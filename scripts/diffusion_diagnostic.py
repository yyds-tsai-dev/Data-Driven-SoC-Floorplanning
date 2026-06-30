#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FLOORSET = ROOT / "FloorSet"
CONTEST = FLOORSET / "iccad2026contest"
for path in (SRC, FLOORSET, CONTEST):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from floorset_arch.diffusion.concretize import (  # noqa: E402
    concretize_diffusion_prior,
    placement_from_tensor_candidate,
)
from floorset_arch.diffusion.diagnostics import (  # noqa: E402
    build_diffusion_diagnostic_report,
    placement_from_fp_sol,
    plot_diffusion_diagnostic,
    xywh_from_fp_sol,
)
from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs  # noqa: E402
from floorset_arch.diffusion.ranking import select_tensor_shortlist  # noqa: E402
from floorset_arch.diffusion.sampling import (  # noqa: E402
    diffusion_guidance_config_from_env,
    load_diffusion_checkpoint,
    sample_diffusion_prior,
)
from floorset_arch.models import SolverConfig  # noqa: E402
from floorset_arch.parser import parse_instance  # noqa: E402
from floorset_arch.repair import repair_placement  # noqa: E402


def _load_evaluator_module():
    path = ROOT / "scripts" / "iccad2026_evaluate.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator module from {path}")
    spec.loader.exec_module(module)
    return module


def _target_positions_from_fp_sol(
    fp_sol: torch.Tensor,
    constraints: torch.Tensor,
    block_count: int,
) -> torch.Tensor:
    target_positions = torch.full((block_count, 4), -1.0)
    fp = xywh_from_fp_sol(fp_sol, block_count)
    cons = torch.as_tensor(constraints).detach().cpu().float()
    for block in range(min(block_count, fp.shape[0], cons.shape[0])):
        x, y, width, height = [float(value) for value in fp[block].tolist()]
        is_fixed = cons.shape[1] > 0 and bool(cons[block, 0].item())
        is_preplaced = cons.shape[1] > 1 and bool(cons[block, 1].item())
        if is_preplaced:
            target_positions[block] = torch.tensor([x, y, width, height])
        elif is_fixed:
            target_positions[block, 2] = width
            target_positions[block, 3] = height
    return target_positions


def _load_validation_case(test_id: int, data_path: Path):
    evaluator = _load_evaluator_module()
    dataset = evaluator.FloorplanDatasetLiteTest(str(data_path))
    sample = dataset[int(test_id)]
    inputs, labels = sample["input"], sample["label"]
    area_targets, b2b, p2b, pins, constraints = inputs
    fp_sol, _metrics = labels
    block_count = int((area_targets != -1).sum().item())
    target_positions = _target_positions_from_fp_sol(fp_sol, constraints, block_count)
    inst = parse_instance(
        block_count,
        area_targets,
        b2b,
        p2b,
        pins,
        constraints,
        target_positions,
    )
    return inst, fp_sol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw raw diffusion, repaired diffusion, and official golden placements."
    )
    parser.add_argument("--test-id", type=int, default=95)
    parser.add_argument("--checkpoint", default=os.environ.get("FLOORSET_DIFFUSION_CHECKPOINT", ""))
    parser.add_argument("--data-path", default=str(FLOORSET))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "debug" / "diffusion_diagnostics"))
    parser.add_argument("--samples", type=int, default=int(os.environ.get("FLOORSET_DIFFUSION_SAMPLES", "8")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("FLOORSET_DIFFUSION_STEPS", "16")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("FLOORSET_DIFFUSION_SEED", "0")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("FLOORSET_DIFFUSION_TOPK", "4")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.checkpoint:
        raise SystemExit("--checkpoint or FLOORSET_DIFFUSION_CHECKPOINT is required")
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    inst, fp_sol = _load_validation_case(args.test_id, Path(args.data_path))
    graph = build_diffusion_graph_inputs(inst, device=torch.device("cpu"))
    model = load_diffusion_checkpoint(checkpoint, graph, map_location="cpu")
    prior = sample_diffusion_prior(
        model,
        graph,
        samples=args.samples,
        steps=args.steps,
        seed=args.seed,
        guidance_config=diffusion_guidance_config_from_env(),
    )
    batch = concretize_diffusion_prior(inst, prior)
    selected = select_tensor_shortlist(batch, top_k=max(1, args.top_k))
    raw_index = int(selected[0].detach().cpu().item())
    raw = placement_from_tensor_candidate(inst, batch, raw_index)
    repaired = repair_placement(inst, raw, SolverConfig())
    golden = placement_from_fp_sol(fp_sol, inst.block_count)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"case_{int(args.test_id)}_sample_{raw_index}"
    image_path = plot_diffusion_diagnostic(
        inst,
        raw,
        repaired,
        golden,
        out_dir / f"{stem}.png",
        case_id=args.test_id,
    )
    report = build_diffusion_diagnostic_report(inst, raw, repaired, golden, case_id=args.test_id)
    report["checkpoint"] = str(checkpoint)
    report["sample_index"] = raw_index
    report_path = out_dir / f"{stem}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {image_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
