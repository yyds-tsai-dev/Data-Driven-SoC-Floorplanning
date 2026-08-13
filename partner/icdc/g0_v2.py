"""Fail-closed G0-v2 two-slot teacher oracle.

Dense training ``fp_sol`` geometry is deliberately confined to the caller's
per-case scope.  Public results contain scalar audit evidence and sparse
topology labels only.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

import torch

from . import engine
from .topology_data import TopologyLabel
from .topology_prior import extract_sparse_label, pin_feasible_then_exact_tfdl


HARD_GAIN = 0.0181504738793652
TARGET_GAIN = 0.0261247299384228


@dataclass(frozen=True)
class G0CaseResult:
    instance_id: str
    n: int
    sample_seed: int
    base_cost: float
    teacher_cost: float
    teacher_candidate_cost: Optional[float]
    winner: Literal["production-base", "transient-fp-exact-tfdl"]
    teacher_status: str
    sparse_label: Optional[TopologyLabel]
    base_hard_audit: Mapping[str, bool]
    teacher_hard_audit: Mapping[str, bool]


@dataclass(frozen=True)
class G0Summary:
    registered_count: int
    result_count: int
    population_sha256: str
    denominator: float
    base_h: Optional[float]
    teacher_h: Optional[float]
    delta_h: Optional[float]
    authorizing: bool
    terminal_state: str


def transient_fp_xywh(
    source: Sequence[torch.Tensor], row: int, n: int
) -> torch.Tensor:
    """Return one verified training fp row in exact CPU-f64 ``(x,y,w,h)``."""
    if not isinstance(source, (list, tuple)) or len(source) != 7:
        raise ValueError("source schema")
    if type(row) is not int or row < 0:
        raise ValueError("row")
    if type(n) is not int or n <= 0:
        raise ValueError("block count")
    fp_sol = source[5]
    if (
        not isinstance(fp_sol, torch.Tensor)
        or fp_sol.ndim != 3
        or fp_sol.shape[-1] != 4
        or row >= fp_sol.shape[0]
    ):
        raise ValueError("row")
    if n > fp_sol.shape[1]:
        raise ValueError("block count")
    raw = fp_sol[row, :n].to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(raw).all()) or not bool((raw[:, :2] > 0).all()):
        raise ValueError("fp values")
    converted = torch.stack(
        (raw[:, 2], raw[:, 3], raw[:, 0], raw[:, 1]), dim=1
    ).contiguous()
    if converted.shape != (n, 4):
        raise ValueError("block count")
    return converted


def _case_tensors(case: Mapping[str, Any]) -> tuple[
    int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]:
    if not isinstance(case, Mapping):
        raise ValueError("case")
    n = case.get("n")
    if type(n) is not int or n <= 0:
        raise ValueError("case n")

    def matrix(name: str, width: int, dtype: torch.dtype) -> torch.Tensor:
        value = case.get(name)
        if not isinstance(value, (list, tuple)):
            raise ValueError(name)
        if not value:
            return torch.empty((0, width), dtype=dtype, device="cpu")
        out = torch.as_tensor(value, dtype=dtype, device="cpu")
        if out.ndim != 2 or out.shape[1] != width:
            raise ValueError(name)
        if torch.is_floating_point(out) and not bool(torch.isfinite(out).all()):
            raise ValueError(name)
        return out.contiguous()

    area = torch.as_tensor(case.get("area"), dtype=torch.float64, device="cpu")
    cons = matrix("cons", 5, torch.int64)
    tp = matrix("tp", 4, torch.float64)
    b2b = matrix("b2b", 3, torch.float64)
    p2b = matrix("p2b", 3, torch.float64)
    pins = matrix("pins", 2, torch.float64)
    if area.shape != (n,) or cons.shape != (n, 5) or tp.shape != (n, 4):
        raise ValueError("case shape")
    if not bool(torch.isfinite(area).all()) or not bool((area > 0).all()):
        raise ValueError("area")
    return n, area.contiguous(), cons, tp, b2b, p2b, pins


def _rects(value: Any, n: int, field: str) -> torch.Tensor:
    try:
        out = torch.as_tensor(value, dtype=torch.float64, device="cpu").contiguous()
    except Exception as exc:
        raise ValueError(field) from exc
    if (
        out.shape != (n, 4)
        or not bool(torch.isfinite(out).all())
        or not bool((out[:, 2:] > 0).all())
    ):
        raise ValueError(field)
    return out


def _hard_audit(
    rects: torch.Tensor,
    area: torch.Tensor,
    cons: torch.Tensor,
    tp: torch.Tensor,
) -> dict[str, bool]:
    raw = engine.verify_hard_legal(
        rects.numpy(), area.numpy(), cons.numpy(), tp.numpy()
    )
    if (
        not isinstance(raw, Mapping)
        or not raw
        or any(not isinstance(key, str) or not key for key in raw)
        or any(type(value) is not bool for value in raw.values())
    ):
        raise ValueError("hard audit")
    return dict(raw)


def _official_cost(
    scorer: Any,
    rects: torch.Tensor,
    case: Mapping[str, Any],
    area: torch.Tensor,
    cons: torch.Tensor,
    tp: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    pins: torch.Tensor,
) -> tuple[bool, Optional[float]]:
    try:
        result = scorer.evaluate_solution(
            {"positions": rects.tolist(), "runtime": 1.0},
            {
                "hpwl_baseline": float(case["hpwl_ref"]),
                "area_baseline": float(case["area_ref"]),
            },
            cons.tolist(),
            b2b,
            p2b,
            pins,
            area,
            target_positions=tp.tolist(),
            median_runtime=1.0,
        )
    except Exception as exc:
        raise RuntimeError("official scorer") from exc
    feasible = getattr(result, "is_feasible", None)
    value = getattr(result, "cost_no_runtime", None)
    if type(feasible) is not bool:
        raise ValueError("official feasible")
    if not feasible:
        return False, None
    if (
        not isinstance(value, numbers.Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("official cost")
    return True, float(value)


def evaluate_case(
    base: Any,
    fp_seed: Any,
    case: Mapping[str, Any],
    scorer: Any,
    *,
    sample_seed: int,
    admit: Callable[[torch.Tensor, Mapping[str, Any]], Any] = (
        pin_feasible_then_exact_tfdl
    ),
) -> G0CaseResult:
    """Evaluate the frozen production base and one transient-fp teacher slot."""
    if type(sample_seed) is not int or isinstance(sample_seed, bool):
        raise ValueError("sample seed")
    instance_id = case.get("instance_id") if isinstance(case, Mapping) else None
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("instance id")
    n, area, cons, tp, b2b, p2b, pins = _case_tensors(case)
    base_rects = _rects(base, n, "base")
    teacher_seed = _rects(fp_seed, n, "fp seed")

    base_hard = _hard_audit(base_rects, area, cons, tp)
    if not all(base_hard.values()):
        raise ValueError("base hard audit")
    base_feasible, base_cost = _official_cost(
        scorer, base_rects, case, area, cons, tp, b2b, p2b, pins
    )
    if not base_feasible or base_cost is None:
        raise ValueError("base unavailable")

    teacher_legal: Optional[torch.Tensor] = None
    teacher_hard: dict[str, bool] = {}
    teacher_candidate_cost: Optional[float] = None
    teacher_status = "admission_failed"
    try:
        admitted = admit(teacher_seed.clone(), case)
    except Exception:
        admitted = None
    if isinstance(admitted, (tuple, list)) and len(admitted) == 2:
        legal_value, drift_value = admitted
        try:
            legal = _rects(legal_value, n, "teacher legal")
            drift = torch.as_tensor(
                drift_value, dtype=torch.float64, device="cpu"
            )
            if drift.shape != (n, 2) or not bool(torch.isfinite(drift).all()):
                raise ValueError("teacher drift")
            if int(torch.count_nonzero(drift).item()) != 0:
                raise ValueError("teacher drift")
            teacher_hard = _hard_audit(legal, area, cons, tp)
            if not all(teacher_hard.values()):
                teacher_status = "hard_audit_failed"
            else:
                feasible, teacher_candidate_cost = _official_cost(
                    scorer, legal, case, area, cons, tp, b2b, p2b, pins
                )
                if not feasible:
                    teacher_status = "official_infeasible"
                    teacher_candidate_cost = None
                else:
                    teacher_legal = legal
                    teacher_status = "candidate"
        except RuntimeError:
            teacher_status = "official_evaluator_error"
        except (TypeError, ValueError):
            if teacher_status == "admission_failed":
                teacher_status = "hard_audit_failed"

    winner = "production-base"
    winner_rects = base_rects
    teacher_cost = base_cost
    if (
        teacher_legal is not None
        and teacher_candidate_cost is not None
        and (teacher_candidate_cost, 1, "transient-fp-exact-tfdl")
        < (base_cost, 0, "production-base")
    ):
        winner = "transient-fp-exact-tfdl"
        winner_rects = teacher_legal
        teacher_cost = teacher_candidate_cost
        teacher_status = "winner"
    elif teacher_legal is not None and teacher_candidate_cost is not None:
        teacher_status = "not_improved"

    label = extract_sparse_label(
        winner_rects,
        case,
        instance_id,
        sample_seed,
        teacher_cost,
        base_cost,
    )
    return G0CaseResult(
        instance_id=instance_id,
        n=n,
        sample_seed=sample_seed,
        base_cost=base_cost,
        teacher_cost=teacher_cost,
        teacher_candidate_cost=teacher_candidate_cost,
        winner=winner,
        teacher_status=teacher_status,
        sparse_label=label,
        base_hard_audit=base_hard,
        teacher_hard_audit=teacher_hard,
    )


def canonical_case_record(result: G0CaseResult) -> bytes:
    if type(result) is not G0CaseResult:
        raise ValueError("case result")
    payload = {
        "instance_id": result.instance_id,
        "n": result.n,
        "sample_seed": result.sample_seed,
        "base_cost": result.base_cost,
        "teacher_cost": result.teacher_cost,
        "teacher_candidate_cost": result.teacher_candidate_cost,
        "winner": result.winner,
        "teacher_status": result.teacher_status,
        "base_hard_audit": dict(result.base_hard_audit),
        "teacher_hard_audit": dict(result.teacher_hard_audit),
    }
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


class _Kahan:
    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        adjusted = value - self.correction
        updated = self.total + adjusted
        self.correction = (updated - self.total) - adjusted
        self.total = updated
        if not math.isfinite(self.total):
            raise ValueError("population overflow")


class PopulationAccumulator:
    """Bound a streamed population and compute its exact declared statistic."""

    def __init__(self, *, authorizing: bool = True) -> None:
        self.authorizing = bool(authorizing)
        self._registered: dict[str, tuple[int, float]] = {}
        self._results: set[str] = set()
        self._population_hash = hashlib.sha256()
        self._denominator = _Kahan()
        self._base = _Kahan()
        self._teacher = _Kahan()

    def register(self, instance_id: str, n: int) -> None:
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("instance id")
        if instance_id in self._registered:
            raise ValueError("duplicate population identity")
        if type(n) is not int or n < 0:
            raise ValueError("n")
        try:
            weight = math.exp(n / 12.0)
        except OverflowError as exc:
            raise ValueError("weight") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("weight")
        line = json.dumps(
            {"instance_id": instance_id, "n": n, "weight": weight},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
        self._population_hash.update(line)
        self._registered[instance_id] = (n, weight)
        self._denominator.add(weight)

    def add(self, result: G0CaseResult) -> None:
        if type(result) is not G0CaseResult:
            raise ValueError("case result")
        registered = self._registered.get(result.instance_id)
        if registered is None:
            raise ValueError("registered identity")
        if result.instance_id in self._results:
            raise ValueError("duplicate result")
        n, weight = registered
        if result.n != n:
            raise ValueError("n mismatch")
        for value in (result.base_cost, result.teacher_cost):
            if (
                not isinstance(value, numbers.Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError("cost")
        self._base.add(float(result.base_cost) * weight)
        self._teacher.add(float(result.teacher_cost) * weight)
        self._results.add(result.instance_id)

    def finish(self, *, complete: bool) -> G0Summary:
        registered_count = len(self._registered)
        result_count = len(self._results)
        population_sha = self._population_hash.hexdigest()
        denominator = self._denominator.total
        if (
            not complete
            or registered_count == 0
            or result_count != registered_count
            or not math.isfinite(denominator)
            or denominator <= 0
        ):
            return G0Summary(
                registered_count,
                result_count,
                population_sha,
                denominator,
                None,
                None,
                None,
                self.authorizing,
                "KILLED_LEGALITY_OR_COVERAGE",
            )
        base_h = self._base.total / denominator
        teacher_h = self._teacher.total / denominator
        delta_h = base_h - teacher_h
        if not all(math.isfinite(value) for value in (base_h, teacher_h, delta_h)):
            raise ValueError("population result")
        if not self.authorizing:
            terminal = "NON_AUTHORIZING_TRACER"
        elif teacher_h > 1.5:
            terminal = "KILLED_TEACHER_GT_1_5"
        elif delta_h < HARD_GAIN:
            terminal = "STOP_HARD_GAIN_MISSED"
        elif delta_h < TARGET_GAIN:
            terminal = "TARGET_GAIN_MISSED_NO_TRAINING_AUTHORITY"
        else:
            terminal = "TARGET_GAIN_MET"
        return G0Summary(
            registered_count,
            result_count,
            population_sha,
            denominator,
            base_h,
            teacher_h,
            delta_h,
            self.authorizing,
            terminal,
        )
