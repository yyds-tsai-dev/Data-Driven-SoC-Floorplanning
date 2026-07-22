#!/usr/bin/env python3
"""Candidate-only, synchronized Direct-DDIM versus Flow probe.

The probe intentionally measures only conditioned candidate generation.  It
does not run the production legalizer or use validation labels for fitting or
selection.  The label payload is accessed only by a narrow adapter that
returns evaluator-visible hard anchors (preplaced xywh, fixed wh); it is then
discarded.  Each requested ``case`` is the official
``LiteTensorDataTest/config_<case>`` input, with one fixed seed protocol and
the shared xyaspect rectangle decoder for every matrix row.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
for import_path in (ROOT / "src", ROOT / "FloorSet", ROOT / "FloorSet" / "iccad2026contest", ROOT / "partner"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


SEED = 20260723
REQUIRED_ROW_FIELDS = frozenset(
    {
        "case_id",
        "model",
        "solver",
        "steps",
        "nfe",
        "samples",
        "seed",
        "cold_load_s",
        "warm_latency_s",
        "peak_memory_bytes",
        "finite",
        "anchor_exact",
        "anchor_max_error",
        "raw_overlap",
        "raw_hpwl_proxy",
        "raw_boundary_violations",
        "raw_group_violations",
        "raw_mib_violations",
        "best_of_k",
        "candidates",
        "repair",
    }
)
REQUIRED_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_index",
        "anchor_exact",
        "anchor_max_error",
        "raw_overlap",
        "raw_hpwl_proxy",
        "raw_boundary_violations",
        "raw_group_violations",
        "raw_mib_violations",
    }
)
_CANDIDATE_NUMERIC_FIELDS = REQUIRED_CANDIDATE_FIELDS - {"candidate_index", "anchor_exact"}


def build_matrix(flow_steps: Iterable[int], solvers: Iterable[str]) -> list[tuple[str, str, int, int]]:
    """Return fixed control plus Flow rows as (model, solver, steps, exact_nfe)."""
    steps = list(flow_steps)
    solver_names = list(solvers)
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("--flow-steps must contain positive integers")
    invalid = sorted(set(solver_names) - {"euler", "heun"})
    if not solver_names or invalid:
        raise ValueError(f"--solvers must contain only euler or heun; invalid={invalid}")
    rows = [("direct", "ddim", 50, 50)]
    for solver in solver_names:
        for step in steps:
            rows.append(("flow", solver, step, step if solver == "euler" else 2 * step))
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True, help="Official config_<case> identifiers.")
    parser.add_argument("--samples", type=int, required=True, help="K candidates per row.")
    parser.add_argument("--flow-steps", type=int, nargs="+", required=True)
    parser.add_argument("--solvers", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT / "FloorSet")
    return parser.parse_args(argv)


def validate_flow_checkpoint(checkpoint: Mapping[str, Any]) -> str:
    """Reject non-flow payloads before their weights reach a model."""
    from flow_train_claude import checkpoint_method

    return checkpoint_method(checkpoint)


def timed_call(
    call: Callable[[], Any],
    *,
    device_type: str,
    synchronize: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Any, float]:
    """Run one warm sampling call with CUDA synchronization bracketing it."""
    sync = synchronize
    if device_type == "cuda" and sync is None:
        sync = torch.cuda.synchronize
    if sync is not None:
        sync()
    started = clock()
    result = call()
    if sync is not None:
        sync()
    return result, clock() - started


def validate_candidate(candidate: Mapping[str, Any], *, label: str = "candidate") -> None:
    if not isinstance(candidate, Mapping):
        raise ValueError(f"{label} must be a mapping")
    missing = sorted(REQUIRED_CANDIDATE_FIELDS - set(candidate))
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(missing)}")
    index = candidate["candidate_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"{label} candidate_index must be a non-negative integer")
    if not isinstance(candidate["anchor_exact"], bool):
        raise ValueError(f"{label} anchor_exact must be boolean")
    for field in _CANDIDATE_NUMERIC_FIELDS:
        value = candidate[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{label} {field} must be a finite number")


def validate_row(row: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_ROW_FIELDS - set(row))
    if missing:
        raise ValueError(f"missing required row fields: {', '.join(missing)}")
    if not isinstance(row["candidates"], list) or not row["candidates"]:
        raise ValueError("row candidates must be a non-empty list")
    if not isinstance(row["best_of_k"], Mapping):
        raise ValueError("row best_of_k must be a mapping")
    for candidate in row["candidates"]:
        validate_candidate(candidate)
    validate_candidate(row["best_of_k"], label="best_of_k")


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[float, int, float, int]:
    overlap = float(candidate["raw_overlap"])
    soft = sum(
        int(candidate[key])
        for key in ("raw_boundary_violations", "raw_group_violations", "raw_mib_violations")
    )
    hpwl = float(candidate["raw_hpwl_proxy"])
    index = int(candidate["candidate_index"])
    if not all(math.isfinite(value) for value in (overlap, hpwl)):
        raise ValueError("candidate metric is non-finite")
    return (overlap, soft, hpwl, index)


def best_of_k(candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pick a deterministic raw candidate: hard overlap, soft, HPWL, index."""
    records = [dict(candidate) for candidate in candidates]
    if not records:
        raise ValueError("cannot select best-of-K from no candidates")
    return min(records, key=_candidate_key)


def _require_checkpoint(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} checkpoint is missing: {resolved}")
    return resolved


def _load_model(path: Path, *, method: str | None, device: torch.device):
    """Load a Direct-compatible model, validating Flow metadata first."""
    from direct_model_claude import DirectDenoiser, DirectModelConfig, EMA

    started = time.perf_counter()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping: {path}")
    if method == "flow":
        validate_flow_checkpoint(payload)
    config_payload = payload.get("model_config")
    if not isinstance(config_payload, Mapping):
        raise ValueError(f"checkpoint has no model_config: {path}")
    config = DirectModelConfig(
        **{key: value for key, value in config_payload.items() if key in DirectModelConfig.__dataclass_fields__}
    )
    if config.z_repr != "xyaspect":
        raise ValueError(f"checkpoint must use xyaspect decoder, got {config.z_repr!r}: {path}")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint has no model weights: {path}")
    model = DirectDenoiser(config).to(device)
    model.load_state_dict(state)
    if "ema" in payload:
        # Preserve the Direct inference contract: EMA, when present, is sampled.
        ema = EMA(model)
        ema.load_state_dict(payload["ema"])
        ema.copy_to(model)
    model.eval()
    return model, config, time.perf_counter() - started


def evaluator_visible_target_positions(
    label_positions: torch.Tensor,
    constraints: torch.Tensor,
    *,
    block_count: int,
) -> torch.Tensor:
    """Apply the ContestEvaluator's solve-visible hard-anchor mask exactly."""
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    positions = torch.as_tensor(label_positions).detach().cpu().float().reshape(-1, 4)
    constraint_tensor = torch.as_tensor(constraints).detach().cpu()
    if positions.shape[0] < block_count or constraint_tensor.shape[0] < block_count:
        raise ValueError("hard-anchor inputs are shorter than block_count")
    anchors = torch.full((block_count, 4), -1.0)
    columns = constraint_tensor.shape[1] if constraint_tensor.dim() > 1 else 0
    for block in range(block_count):
        is_fixed = columns > 0 and constraint_tensor[block, 0] != 0
        is_preplaced = columns > 1 and constraint_tensor[block, 1] != 0
        if is_preplaced:
            anchors[block] = positions[block]
        elif is_fixed:
            anchors[block, 2:] = positions[block, 2:]
    return anchors


def _load_evaluator_exposed_hard_anchors(
    label_path: Path,
    constraints: torch.Tensor,
    block_count: int,
) -> torch.Tensor:
    """Read a label only to derive and immediately mask evaluator-visible anchors."""
    from floorset_arch.diffusion.diagnostics import xywh_from_fp_sol

    label_payload = torch.load(label_path, map_location="cpu", weights_only=False)[0]
    full_positions = xywh_from_fp_sol(label_payload[1], block_count)
    anchors = evaluator_visible_target_positions(full_positions, constraints, block_count=block_count)
    del full_positions
    del label_payload
    return anchors


def _load_official_case(case_id: int, data_path: Path):
    """Read exactly config_<case>/litedata_1, never a label-selected index."""
    if not isinstance(case_id, int) or case_id < 21 or case_id > 120:
        raise ValueError(f"case must be an official config id in [21, 120], got {case_id!r}")
    config_dir = data_path / "LiteTensorDataTest" / f"config_{case_id}"
    input_path = config_dir / "litedata_1.pth"
    label_path = config_dir / "litelabel_1.pth"
    if not input_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"missing official validation input for config_{case_id}: {config_dir}")
    inputs = torch.load(input_path, map_location="cpu", weights_only=False)[0]
    area_and_constraints, b2b, p2b, pins = inputs
    area = area_and_constraints[:, 0].float()
    constraints = area_and_constraints[:, 1:].float()
    block_count = int((area != -1).sum().item())
    if block_count <= 0:
        raise ValueError(f"config_{case_id} has no active blocks")

    from floorset_arch.parser import parse_instance

    evaluator_anchors = _load_evaluator_exposed_hard_anchors(label_path, constraints, block_count)
    instance = parse_instance(block_count, area, b2b, p2b, pins, constraints, evaluator_anchors)
    return area, b2b.float(), p2b.float(), pins.float(), constraints, evaluator_anchors, instance


def _expand_condition(condition: Mapping[str, torch.Tensor], samples: int) -> dict[str, torch.Tensor]:
    return {
        key: value.expand(samples, *value.shape[1:]).contiguous()
        for key, value in condition.items()
    }


def _build_condition(case, config, device: torch.device, samples: int):
    from direct_model_claude import known_z_channels
    from direct_train_claude import fast_condition
    from diffusion_data import layout_scale

    area, b2b, p2b, pins, constraints, target_positions, _instance = case
    area_b = area.unsqueeze(0).to(device)
    constraints_b = constraints.unsqueeze(0).to(device)
    positions_b = target_positions.unsqueeze(0).to(device)
    condition = fast_condition(
        area_b,
        b2b.unsqueeze(0).to(device),
        p2b.unsqueeze(0).to(device),
        pins.unsqueeze(0).to(device),
        constraints_b,
        positions_b,
        relation_feat_dim=config.relation_feat_dim,
        node_feat_dim=config.node_feat_dim,
    )
    if condition["node_feat"].shape[-1] != config.node_feat_dim:
        raise ValueError("condition node feature dimension is incompatible with checkpoint")
    if condition["rel_feat"].shape[-1] != config.relation_feat_dim:
        raise ValueError("condition relation feature dimension is incompatible with checkpoint")
    scale = layout_scale(area_b)
    z_known, known_mask = known_z_channels(area_b, constraints_b, positions_b, scale)
    return (
        _expand_condition(condition, samples),
        area_b.expand(samples, -1),
        constraints_b.expand(samples, -1, -1),
        positions_b.expand(samples, -1, -1),
        z_known.expand(samples, -1, -1),
        known_mask.expand(samples, -1, -1),
    )


def _placement_metrics_for_candidates(instance, rectangles: torch.Tensor) -> list[dict[str, Any]]:
    from floorset_arch.diagnostics import placement_metrics
    from floorset_arch.models import Placement, Rect

    candidates = []
    for index, candidate in enumerate(rectangles.detach().cpu().tolist()):
        placement = Placement({block: Rect(*map(float, rect)) for block, rect in enumerate(candidate)})
        metrics = placement_metrics(instance, placement)
        anchor_error = 0.0
        for block, target in instance.target_rects.items():
            actual = placement.rects[block]
            expected = target.as_tuple()
            observed = actual.as_tuple()
            channels = range(4) if block in instance.preplaced else range(2, 4)
            anchor_error = max(anchor_error, *(abs(observed[channel] - expected[channel]) for channel in channels))
        candidates.append(
            {
                "candidate_index": index,
                "anchor_exact": anchor_error <= 1e-6,
                "anchor_max_error": anchor_error,
                "raw_overlap": int(metrics["overlap_count"]),
                "raw_hpwl_proxy": float(metrics["hpwl_proxy"]),
                "raw_boundary_violations": int(metrics["boundary_violations"]),
                "raw_group_violations": int(metrics["group_violations"]),
                "raw_mib_violations": int(metrics["mib_violations"]),
            }
        )
    return candidates


def _sample_row(
    *,
    case_id: int,
    case,
    model,
    config,
    matrix_row: tuple[str, str, int, int],
    samples: int,
    cold_load_s: float,
    device: torch.device,
) -> dict[str, Any]:
    from diffusion_data import z_to_rectangles
    from diffusion_model import DiffusionSchedule
    from direct_model_claude import sample_direct
    from flow_matching_claude import sample_flow

    model_name, solver, steps, expected_nfe = matrix_row
    condition, area, constraints, positions, z_known, known_mask = _build_condition(case, config, device, samples)
    generator = torch.Generator(device=device).manual_seed(SEED)
    if model_name == "direct":
        schedule = DiffusionSchedule(config.timesteps, device=device)

        def sample():
            return sample_direct(model, condition, schedule, steps=50, generator=generator, z_known=z_known, known_mask=known_mask)

    else:

        def sample():
            return sample_flow(model, condition, steps=steps, solver=solver, generator=generator, z_known=z_known, known_mask=known_mask)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sample_result, latency_s = timed_call(sample, device_type=device.type)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    if model_name == "flow":
        if sample_result.nfe != expected_nfe:
            raise RuntimeError(f"Flow sampler NFE mismatch: expected {expected_nfe}, got {sample_result.nfe}")
        z = sample_result.z
    else:
        z = sample_result
    if not torch.isfinite(z).all():
        raise ValueError(f"non-finite {model_name} sample for config_{case_id}, {solver}/{steps}")
    rectangles = z_to_rectangles(z, area, target_positions=positions, constraints=constraints, z_repr="xyaspect")
    candidates = _placement_metrics_for_candidates(case[-1], rectangles)
    best = best_of_k(candidates)
    row = {
        "case_id": case_id,
        "model": model_name,
        "solver": solver,
        "steps": steps,
        "nfe": expected_nfe,
        "samples": samples,
        "seed": SEED,
        "cold_load_s": cold_load_s,
        "warm_latency_s": latency_s,
        "peak_memory_bytes": peak_memory,
        "finite": True,
        "anchor_exact": bool(best["anchor_exact"]),
        "anchor_max_error": float(best["anchor_max_error"]),
        "raw_overlap": int(best["raw_overlap"]),
        "raw_hpwl_proxy": float(best["raw_hpwl_proxy"]),
        "raw_boundary_violations": int(best["raw_boundary_violations"]),
        "raw_group_violations": int(best["raw_group_violations"]),
        "raw_mib_violations": int(best["raw_mib_violations"]),
        "best_of_k": best,
        "candidates": candidates,
        "repair": {
            "available": False,
            "reason": "No bounded partner-only repair seam is invoked; raw candidate metrics only.",
        },
    }
    validate_row(row)
    return row


def _warm_model(model, config, case, *, model_name: str, samples: int, device: torch.device) -> None:
    """One unmeasured call per loaded model, preserving the same condition/decoder path."""
    warm_matrix = ("direct", "ddim", 50, 50) if model_name == "direct" else ("flow", "euler", 1, 1)
    _sample_row(
        case_id=-1,
        case=case,
        model=model,
        config=config,
        matrix_row=warm_matrix,
        samples=samples,
        cold_load_s=0.0,
        device=device,
    )


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = f"{row['model']}:{row['solver']}:steps={row['steps']}:nfe={row['nfe']}"
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, group in groups.items():
        summary[key] = {}
        for metric in ("warm_latency_s", "raw_overlap", "raw_hpwl_proxy"):
            values = sorted(float(row[metric]) for row in group)
            last = len(values) - 1
            summary[key][f"{metric}_median"] = values[last // 2] if len(values) % 2 else (values[last // 2] + values[last // 2 + 1]) / 2
            summary[key][f"{metric}_p90"] = values[min(last, math.ceil(0.9 * len(values)) - 1)]
            summary[key][f"{metric}_max"] = values[-1]
    return summary


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    matrix = build_matrix(args.flow_steps, args.solvers)
    cases = list(args.cases)
    if len(set(cases)) != len(cases):
        raise ValueError("--cases must not contain duplicates")
    direct_path = _require_checkpoint(args.direct_checkpoint, "Direct")
    flow_path = _require_checkpoint(args.flow_checkpoint, "Flow")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direct_model, direct_config, direct_load_s = _load_model(direct_path, method="direct", device=device)
    flow_model, flow_config, flow_load_s = _load_model(flow_path, method="flow", device=device)
    loaded_cases = {case_id: _load_official_case(case_id, args.data_path) for case_id in cases}
    _warm_model(direct_model, direct_config, loaded_cases[cases[0]], model_name="direct", samples=args.samples, device=device)
    _warm_model(flow_model, flow_config, loaded_cases[cases[0]], model_name="flow", samples=args.samples, device=device)

    rows = []
    for case_id in cases:
        for matrix_row in matrix:
            model_name = matrix_row[0]
            model, config, load_s = (
                (direct_model, direct_config, direct_load_s)
                if model_name == "direct"
                else (flow_model, flow_config, flow_load_s)
            )
            rows.append(
                _sample_row(
                    case_id=case_id,
                    case=loaded_cases[case_id],
                    model=model,
                    config=config,
                    matrix_row=matrix_row,
                    samples=args.samples,
                    cold_load_s=load_s,
                    device=device,
                )
            )
    return {
        "metadata": {
            "device": str(device),
            "seed_protocol": "fixed per row: 20260723",
            "case_resolution": "official LiteTensorDataTest/config_<case>/litedata_1.pth",
            "anchor_inputs": (
                "ContestEvaluator solve-visible hard anchors only: preplaced xywh, fixed wh; "
                "labels are not retained or used for fitting, model/checkpoint selection, or quality metrics."
            ),
            "cold_load_s": {"direct": direct_load_s, "flow": flow_load_s},
            "repair": "not run; no bounded partner-only seam",
        },
        "matrix": [
            {"model": model, "solver": solver, "steps": steps, "nfe": nfe}
            for model, solver, steps, nfe in matrix
        ],
        "rows": rows,
        "summary": _summary(rows),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
