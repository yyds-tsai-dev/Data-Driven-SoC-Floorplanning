"""Heldout sparse-topology transfer audit for a same-shape Direct checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from .qa_contract import preflight_qa_contract, score_provided_local_no_runtime
from .topology_prior import pin_feasible_then_exact_tfdl
from .train_topology_prior import (
    CONTROL_FILE_SHA256,
    FLOW_FILE_SHA256,
    _atomic_bytes,
    _batch,
    _canonical_bytes,
    _fixed_noise,
    _sha256,
    checkpoint_contract,
    energy,
    engine,
    expand_cond,
    frozen_portfolio_contract,
    load_paired_records,
    sample_differentiable,
)


def retained_gain_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("empty audit")
    ids: set[str] = set()
    teacher_terms: list[float] = []
    student_terms: list[float] = []
    weights: list[float] = []
    admitted = 0
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "instance_id", "n", "weight", "base_cost", "teacher_cost",
            "student_cost", "admitted",
        }:
            raise ValueError("audit row")
        instance_id = row["instance_id"]
        if not isinstance(instance_id, str) or not instance_id or instance_id in ids:
            raise ValueError("audit identity")
        ids.add(instance_id)
        if type(row["n"]) is not int or int(row["n"]) <= 0:
            raise ValueError("audit n")
        values = [float(row[key]) for key in (
            "weight", "base_cost", "teacher_cost", "student_cost"
        )]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("audit values")
        if type(row["admitted"]) is not bool:
            raise ValueError("audit admission")
        weight, base, teacher, student = values
        teacher_gain = base - teacher
        if teacher_gain <= 0:
            continue
        weights.append(weight)
        teacher_terms.append(weight * teacher_gain)
        student_terms.append(weight * (base - student))
        admitted += int(row["admitted"])
    denominator = math.fsum(teacher_terms)
    numerator = math.fsum(student_terms)
    if not weights or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("positive teacher gain")
    retained = numerator / denominator
    if not math.isfinite(numerator) or not math.isfinite(retained):
        raise ValueError("retained gain")
    return {
        "eligible_count": len(weights),
        "admitted_count": admitted,
        "teacher_gain_weighted_sum": denominator,
        "student_gain_weighted_sum": numerator,
        "retained_gain_fraction": retained,
        "retained_gain_pass": retained >= 0.75,
    }


@torch.no_grad()
def _decoded_candidates(
    model: torch.nn.Module,
    schedule: Any,
    records: Sequence[Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    samples = 3
    batch = _batch(records, device)
    labels = [record.label for record in records]
    cond = engine.build_cond(batch, model.config)
    known_z, known_mask = engine.known_channels(batch)
    cond_k = expand_cond(cond, samples)
    known_z_k = known_z.repeat_interleave(samples, dim=0)
    known_mask_k = known_mask.repeat_interleave(samples, dim=0)
    batch_k = {
        key: value.repeat_interleave(samples, dim=0)
        for key, value in batch.items()
    }
    noise = _fixed_noise(
        labels,
        samples,
        int(batch["area"].shape[1]),
        int(model.config.z_dim),
        device,
    )
    seed = int.from_bytes(
        hashlib.sha256(
            b"\0".join(str(label.sample_seed).encode("ascii") for label in labels)
        ).digest()[:8],
        "big",
    ) % (2**63 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)
    z = sample_differentiable(
        model,
        cond_k,
        schedule,
        steps=2,
        generator=generator,
        z_known=known_z_k,
        known_mask=known_mask_k,
        grad_steps=0,
        noise=noise,
    )
    rects = energy.decode_rects(
        z, batch_k["area"], batch_k["cons"], batch_k["tp"], batch_k["scale"]
    )
    return rects.view(len(records), samples, rects.shape[1], 4)


def audit(args: argparse.Namespace) -> dict[str, object]:
    candidate_path = Path(args.candidate)
    control_path = Path(args.control_checkpoint)
    flow_path = Path(args.flow_checkpoint)
    if _sha256(control_path) != CONTROL_FILE_SHA256:
        raise ValueError("control checkpoint")
    if _sha256(flow_path) != FLOW_FILE_SHA256:
        raise ValueError("flow checkpoint")
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    control = torch.load(control_path, map_location="cpu", weights_only=False)
    portfolio = frozen_portfolio_contract()
    contract = checkpoint_contract(
        control,
        candidate,
        sampler_method="dpmpp",
        sampler_steps=2,
        candidate_count=6,
        portfolio_contract_sha256=portfolio["sha256"],
    )
    if not contract["ok"]:
        raise ValueError("candidate checkpoint contract")
    records = load_paired_records(
        args.corpus,
        args.labels,
        selection_mod=args.selection_mod,
        selection_namespace="icdc-topology-heldout-audit-v1",
        max_records=args.max_records,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda unavailable")
    from direct_diffusion_model import DirectDenoiser, DirectModelConfig
    from diffusion_model import DiffusionSchedule

    config_fields = DirectModelConfig.__dataclass_fields__
    config = DirectModelConfig(**{
        key: value
        for key, value in candidate["model_config"].items()
        if key in config_fields
    })
    model = DirectDenoiser(config).to(device)
    model.load_state_dict(candidate["ema"], strict=True)
    model.eval()
    schedule = DiffusionSchedule(config.timesteps, device=device)
    qa = preflight_qa_contract(Path(__file__).resolve().parents[2])
    rows: list[dict[str, object]] = []
    for start in range(0, len(records), args.batch):
        group = records[start : start + args.batch]
        decoded = _decoded_candidates(model, schedule, group, device=device)
        for local, record in enumerate(group):
            n = record.label.n
            costs: list[float] = []
            for sample in range(3):
                proposal = decoded[local, sample, :n].detach().cpu().to(torch.float64)
                admitted = pin_feasible_then_exact_tfdl(proposal, record.case)
                if admitted is None:
                    continue
                audit_result = score_provided_local_no_runtime(
                    record.case, admitted[0], qa
                )
                if audit_result.feasible:
                    costs.append(audit_result.cost_no_runtime)
            rows.append({
                "instance_id": record.label.instance_id,
                "n": n,
                "weight": math.exp(n / 12.0),
                "base_cost": float(record.label.base_cost),
                "teacher_cost": float(record.label.teacher_cost),
                "student_cost": min(costs) if costs else 10.0,
                "admitted": bool(costs),
            })
    summary = retained_gain_summary(rows)
    result = {
        "schema": "icdc_topology_heldout_audit_v1",
        "candidate_sha256": _sha256(candidate_path),
        "control_sha256": _sha256(control_path),
        "flow_sha256": _sha256(flow_path),
        "corpus_sha256": _sha256(Path(args.corpus)),
        "labels_sha256": _sha256(Path(args.labels)),
        "selection_mod": args.selection_mod,
        "record_count": len(rows),
        **summary,
        "rows": rows,
    }
    _atomic_bytes(Path(args.out), _canonical_bytes(result, newline=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection-mod", type=int, default=64)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.selection_mod <= 0 or args.batch <= 0:
        raise ValueError("audit config")
    result = audit(args)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))
    return 0 if result["retained_gain_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
