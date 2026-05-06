from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from floorset_arch.features import build_model_inputs
from floorset_arch.nn.model import SimpleGraphFloorplanner
from floorset_arch.nn.postprocess import predictions_to_positions
from floorset_arch.parser import parse_instance
from floorset_arch.training.checkpoint import save_checkpoint
from floorset_arch.training.losses import compute_v1_loss


ROOT = Path(__file__).resolve().parents[3]
FLOORSET = ROOT / "FloorSet"
CONTEST_DIR = FLOORSET / "iccad2026contest"
for path in (FLOORSET, CONTEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from iccad2026_evaluate import get_training_dataloader, get_validation_dataloader  # noqa: E402


def _fp_to_xywh(fp_sol: torch.Tensor, block_count: int) -> torch.Tensor:
    fp = fp_sol[:block_count].float()
    if fp.dim() != 2 or fp.shape[1] != 4:
        raise ValueError("training fp_sol must have shape [N, 4] as [w, h, x, y]")
    return torch.stack([fp[:, 2], fp[:, 3], fp[:, 0], fp[:, 1]], dim=1)


def _polygons_to_xywh(polygons: torch.Tensor, block_count: int) -> torch.Tensor:
    rows = []
    for i in range(block_count):
        poly = polygons[i]
        valid = poly[poly[:, 0] != -1]
        x_min, y_min = valid.min(dim=0).values
        x_max, y_max = valid.max(dim=0).values
        rows.append(torch.stack([x_min, y_min, x_max - x_min, y_max - y_min]))
    return torch.stack(rows, dim=0).float()


def _sample_iterator(args):
    if not args.use_validation_smoke:
        loader = get_training_dataloader(
            data_path=args.data_path,
            batch_size=1,
            num_samples=args.num_samples,
            shuffle=False,
        )
        for batch in loader:
            area, b2b, p2b, pins, constraints, _tree_sol, fp_sol, metrics = batch
            area = area.squeeze(0)
            b2b = b2b.squeeze(0)
            p2b = p2b.squeeze(0)
            pins = pins.squeeze(0)
            constraints = constraints.squeeze(0)
            fp_sol = fp_sol.squeeze(0)
            metrics = metrics.squeeze(0)
            block_count = int((area != -1).sum().item())
            target = _fp_to_xywh(fp_sol, block_count)
            yield area, b2b, p2b, pins, constraints, metrics, block_count, target
        return

    print("using validation data for smoke training")
    loader = get_validation_dataloader(data_path=args.data_path, batch_size=1)
    for idx, batch in enumerate(loader):
        if idx >= args.num_samples:
            break
        inputs, labels = batch
        area, b2b, p2b, pins, constraints = [item.squeeze(0) for item in inputs]
        polygons, metrics = labels
        polygons = polygons.squeeze(0)
        metrics = metrics.squeeze(0)
        block_count = int((area != -1).sum().item())
        target = _polygons_to_xywh(polygons, block_count)
        yield area, b2b, p2b, pins, constraints, metrics, block_count, target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="FloorSet")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="checkpoints/v1.pt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument(
        "--use-validation-smoke",
        action="store_true",
        help="Use the small validation set instead of the official training dataloader. "
        "This is only for local smoke tests when the training archive is unavailable.",
    )
    args = parser.parse_args()

    if args.batch_size != 1:
        print("architecture v1.0 training currently supports --batch-size 1 only", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    model = None
    optimizer = None
    model_config = {"hidden_dim": args.hidden_dim, "layers": args.layers}
    for epoch in range(args.epochs):
        running = 0.0
        count = 0
        for area, b2b, p2b, pins, constraints, metrics, block_count, target in _sample_iterator(args):
            inst = parse_instance(block_count, area, b2b, p2b, pins, constraints, None)
            inputs = build_model_inputs(inst)
            if model is None:
                model = SimpleGraphFloorplanner(
                    input_dim=inputs.block_features.shape[1],
                    hidden_dim=args.hidden_dim,
                    layers=args.layers,
                ).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            inputs.block_features = inputs.block_features.to(device)
            inputs.edge_index = inputs.edge_index.to(device)
            inputs.edge_weight = inputs.edge_weight.to(device)
            inputs.pin_features = inputs.pin_features.to(device)
            inst.area_targets = inst.area_targets.to(device)
            inst.pins_pos = inst.pins_pos.to(device)
            pred = model(inputs)
            positions = predictions_to_positions(inst, pred)
            target = target.to(device)
            loss = compute_v1_loss(positions, target, inst, metrics.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            count += 1
        avg = running / max(count, 1)
        print(f"epoch={epoch + 1} samples={count} loss={avg:.4f}")

    if model is None:
        raise RuntimeError("no training samples were loaded")
    save_checkpoint(args.out, model, model_config, {"num_samples": args.num_samples, "epochs": args.epochs})
    print(f"saved checkpoint to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
