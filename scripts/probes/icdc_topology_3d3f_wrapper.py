#!/usr/bin/env python3
"""Evaluator wrapper for the frozen raw-six 3 Direct + 3 Flow G1 portfolio."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


_REPO = Path(__file__).resolve().parents[2]
for _dependency in (_REPO / "src" / "solver", _REPO / "FloorSet" / "iccad2026contest"):
    if str(_dependency) not in sys.path:
        sys.path.insert(0, str(_dependency))

import column_sa_legalizer as _legalizer  # noqa: E402
import contest_optimizer as _contest  # noqa: E402
from icdc_engine.g1_runtime import (  # noqa: E402
    PortfolioContractError,
    assert_solver_environment,
    environment_identities,
    validate_portfolio_receipt,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class MyOptimizer(_contest.MyOptimizer):
    """Production optimizer with fail-closed, return-value-neutral tracing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert_solver_environment(os.environ)
        self._receipt_path = Path(os.environ["FLOORSET_TOPOLOGY_RECEIPT"])
        self._arm_id = os.environ["FLOORSET_OPAQUE_ARM_ID"]
        if not self._arm_id:
            raise PortfolioContractError("opaque arm id")
        try:
            self._expected_cases = int(os.environ.get(
                "FLOORSET_TOPOLOGY_EXPECTED_CASES", "100"
            ))
        except ValueError as exc:
            raise PortfolioContractError("expected cases") from exc
        if self._expected_cases <= 0:
            raise PortfolioContractError("expected cases")
        self._case_records: list[dict[str, object]] = []
        self._active_trace: dict[str, Any] | None = None
        self._invalid_events: list[str] = []
        self._environment = environment_identities(os.environ)
        super().__init__(*args, **kwargs)
        if self.direct_model is None or self.flow_model is None:
            raise PortfolioContractError("both Direct and Flow models must load")
        atexit.register(self._write_complete_receipt)

    def _receipt(self, complete: bool) -> dict[str, object]:
        return {
            "schema": "icdc_topology_3d3f_arm_receipt_v1",
            "arm_id": self._arm_id,
            "complete": complete,
            "expected_cases": self._expected_cases,
            "environment": self._environment,
            "cases": self._case_records,
            "invalid_events": self._invalid_events,
        }

    def _write_complete_receipt(self) -> None:
        complete = (
            len(self._case_records) == self._expected_cases
            and not self._invalid_events
        )
        _atomic_json(self._receipt_path, self._receipt(complete))

    def _fail_closed(self, events: list[str]) -> None:
        self._invalid_events.extend(events)
        _atomic_json(self._receipt_path, self._receipt(False))
        os._exit(86)

    def _sample_flow_preds(self, *args: Any, **kwargs: Any) -> list[np.ndarray]:
        trace = self._active_trace
        try:
            output = super()._sample_flow_preds(*args, **kwargs)
        except BaseException as exc:
            if trace is not None:
                trace["flow_exception"] = f"{type(exc).__name__}: {exc}"
            raise
        if trace is not None:
            requested = kwargs.get("K", args[7] if len(args) > 7 else None)
            trace["flow_requested"] = requested
            trace["flow_hashes"] = [_array_sha256(item) for item in output]
        return output

    def _sample_direct_raw_preds(self, *args: Any, **kwargs: Any) -> list[np.ndarray]:
        trace = self._active_trace
        if trace is not None:
            trace["sample_calls"] += 1
            trace["requested_K"] = kwargs.get("K", args[7] if len(args) > 7 else None)
            trace["oversample"] = kwargs.get(
                "oversample", args[8] if len(args) > 8 else True
            )
        output = super()._sample_direct_raw_preds(*args, **kwargs)
        if trace is not None:
            flow_hashes = list(trace.get("flow_hashes", []))
            mixed_hashes = [_array_sha256(item) for item in output]
            flow_count = len(flow_hashes)
            direct_count = len(output) - flow_count
            trace["direct_count"] = direct_count
            trace["flow_count"] = flow_count
            trace["post_candidate_count"] = len(output)
            trace["direct_hashes"] = mixed_hashes[:direct_count]
            trace["post_mix_hashes"] = mixed_hashes
            trace["flow_order_ok"] = (
                flow_count > 0 and mixed_hashes[direct_count:] == flow_hashes
            )
        return output

    def _direct_worker(self, *args: Any, **kwargs: Any) -> None:
        if self._active_trace is not None:
            self._active_trace["direct_worker_fallback"] = True
        return super()._direct_worker(*args, **kwargs)

    def solve(self, *args: Any, **kwargs: Any):
        block_count = int(args[0] if args else kwargs["block_count"])
        ordinal = len(self._case_records)
        trace: dict[str, Any] = {
            "sample_calls": 0,
            "gate_calls": [],
            "parallel_calls": 0,
            "parallel_exception": None,
            "outer_fallback": False,
            "direct_worker_fallback": False,
            "flow_exception": None,
            "flow_hashes": [],
        }
        self._active_trace = trace
        original_gate = _contest._direct_seat_ok
        original_parallel = _legalizer._parallel_solve
        original_fallback = _contest._fallback_row

        def traced_gate(n: int, remaining: float) -> bool:
            result = bool(original_gate(n, remaining))
            trace["gate_calls"].append(result)
            trace["pool_ready_at_gate"] = bool(
                _legalizer._POOL is not None and _legalizer._POOL_READY
            )
            return result

        def traced_parallel(*parallel_args: Any, **parallel_kwargs: Any):
            trace["parallel_calls"] += 1
            try:
                return original_parallel(*parallel_args, **parallel_kwargs)
            except BaseException as exc:
                trace["parallel_exception"] = f"{type(exc).__name__}: {exc}"
                raise

        def traced_fallback(*fallback_args: Any, **fallback_kwargs: Any):
            trace["outer_fallback"] = True
            return original_fallback(*fallback_args, **fallback_kwargs)

        _contest._direct_seat_ok = traced_gate
        _legalizer._parallel_solve = traced_parallel
        _contest._fallback_row = traced_fallback
        try:
            result = super().solve(*args, **kwargs)
        finally:
            _contest._direct_seat_ok = original_gate
            _legalizer._parallel_solve = original_parallel
            _contest._fallback_row = original_fallback
            self._active_trace = None

        gate_open = trace["gate_calls"] == [True]
        gate_closed = trace["gate_calls"] == [False]
        portfolio = {
            "case_id": str(ordinal),
            "direct_gate_open": gate_open,
            "pool_ready": bool(trace.get("pool_ready_at_gate", True)),
            "requested_K": int(trace.get("requested_K", 0) or 0),
            "oversample": bool(trace.get("oversample", False)),
            "direct_count": int(trace.get("direct_count", 0)),
            "flow_count": int(trace.get("flow_count", 0)),
            "post_candidate_count": int(trace.get("post_candidate_count", 0)),
            "direct_sampler": "dpmpp",
            "direct_steps": 2,
            "flow_sampler": "euler",
            "flow_steps": 8,
            "flow_exception": trace["flow_exception"],
            "fallback": bool(
                trace["outer_fallback"]
                or trace["direct_worker_fallback"]
                or trace["parallel_exception"]
            ),
        }
        errors: list[str] = []
        if not (gate_open or gate_closed):
            errors.append("direct_gate_trace")
        if gate_open:
            if trace["sample_calls"] != 1:
                errors.append("sample_call_count")
            if trace["parallel_calls"] != 1:
                errors.append("parallel_call_count")
            if not trace.get("flow_order_ok", False):
                errors.append("flow_order")
        elif trace["sample_calls"]:
            errors.append("sample_on_closed_gate")
        try:
            validate_portfolio_receipt(portfolio)
        except PortfolioContractError as exc:
            errors.append(str(exc))

        case_record = {
            "ordinal": ordinal,
            "block_count": block_count,
            "status": "normal_pool_valid" if gate_open else "direct_gate_closed",
            "portfolio": portfolio,
            "direct_hashes": trace.get("direct_hashes", []),
            "flow_hashes": trace.get("flow_hashes", []),
            "post_mix_hashes": trace.get("post_mix_hashes", []),
            "final_layout_sha256": _array_sha256(result),
        }
        self._case_records.append(case_record)
        if errors:
            self._fail_closed([f"case {ordinal}: {error}" for error in errors])
        return result
