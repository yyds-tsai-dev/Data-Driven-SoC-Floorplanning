from __future__ import annotations

from dataclasses import dataclass

import torch

from floorset_arch.diffusion.contracts import DiffusionGraphInputs
from floorset_arch.models import Instance


@dataclass(frozen=True)
class TreeEdges:
    parent: torch.Tensor
    child: torch.Tensor
    side: torch.Tensor


@dataclass(frozen=True)
class MetricsSplit:
    instance_stats: torch.Tensor
    quality_labels: torch.Tensor


@dataclass(frozen=True)
class DiffusionTargets:
    center: torch.Tensor
    log_aspect: torch.Tensor
    pair_axis_label: torch.Tensor
    tree_pair_mask: torch.Tensor
    tree_side_label: torch.Tensor
    instance_stats: torch.Tensor
    quality_labels: torch.Tensor


def parse_tree_sol_edges(tree_sol: torch.Tensor | None, block_count: int) -> TreeEdges:
    device = tree_sol.device if tree_sol is not None else torch.device("cpu")
    empty = torch.empty(0, dtype=torch.long, device=device)
    if tree_sol is None or tree_sol.numel() == 0:
        return TreeEdges(empty, empty, empty)

    rows = torch.as_tensor(tree_sol, device=device).detach()
    if rows.dim() == 1:
        rows = rows.reshape(-1, 3)
    if rows.shape[-1] < 3:
        raise ValueError("tree_sol must have at least 3 columns: parent, child, side")
    rows = rows.reshape(-1, rows.shape[-1])[:, :3].long()
    parent = rows[:, 0]
    child = rows[:, 1]
    side = rows[:, 2]
    valid = (
        (parent >= 0)
        & (parent < block_count)
        & (child >= 0)
        & (child < block_count)
        & (parent != child)
        & (side >= 0)
        & (side <= 1)
    )
    return TreeEdges(parent[valid], child[valid], side[valid])


def split_metrics_sol(metrics_sol: torch.Tensor | None) -> MetricsSplit:
    device = metrics_sol.device if metrics_sol is not None else torch.device("cpu")
    if metrics_sol is None or metrics_sol.numel() == 0:
        empty = torch.empty(0, dtype=torch.float32, device=device)
        return MetricsSplit(empty, empty)

    metrics = torch.as_tensor(metrics_sol, device=device).detach().float().flatten()
    if metrics.numel() < 8:
        quality = metrics[:1].clone()
        stats = metrics[1:].clone()
        return MetricsSplit(stats, quality)
    quality = torch.stack((metrics[0], metrics[-2], metrics[-1]))
    return MetricsSplit(metrics[1:-2].clone(), quality)


def _fp_sol_xywh(fp_sol: torch.Tensor, block_count: int, device: torch.device) -> torch.Tensor:
    sol = torch.as_tensor(fp_sol, device=device).detach().float()
    if sol.dim() == 1:
        sol = sol.reshape(-1, 4)
    if sol.shape[-1] < 4:
        raise ValueError("fp_sol must have at least 4 columns: width, height, x, y")
    sol = sol.reshape(-1, sol.shape[-1])[:block_count, :4]
    if sol.shape[0] != block_count:
        raise ValueError(f"fp_sol must include {block_count} block rows")
    width = sol[:, 0].clamp_min(1e-6)
    height = sol[:, 1].clamp_min(1e-6)
    x = sol[:, 2]
    y = sol[:, 3]
    return torch.stack((x, y, width, height), dim=1)


def _pair_axis_labels(
    xywh: torch.Tensor,
    pair_index: torch.Tensor,
    tree_edges: TreeEdges,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pair_count = int(pair_index.shape[0])
    labels = torch.full((pair_count,), 2, dtype=torch.long, device=pair_index.device)
    tree_mask = torch.zeros(pair_count, dtype=torch.bool, device=pair_index.device)
    tree_side = torch.full((pair_count,), -1, dtype=torch.long, device=pair_index.device)
    tree_lookup = {
        tuple(sorted((int(parent), int(child)))): int(side)
        for parent, child, side in zip(
            tree_edges.parent.detach().cpu().tolist(),
            tree_edges.child.detach().cpu().tolist(),
            tree_edges.side.detach().cpu().tolist(),
            strict=True,
        )
    }
    for idx, (i_raw, j_raw) in enumerate(pair_index.detach().cpu().tolist()):
        i, j = int(i_raw), int(j_raw)
        delta = xywh[j, :2] - xywh[i, :2]
        labels[idx] = 0 if abs(float(delta[0].item())) >= abs(float(delta[1].item())) else 1
        side = tree_lookup.get(tuple(sorted((i, j))))
        if side is not None:
            tree_mask[idx] = True
            tree_side[idx] = side
    return labels, tree_mask, tree_side


def build_diffusion_targets(
    inst: Instance,
    graph_inputs: DiffusionGraphInputs,
    fp_sol: torch.Tensor,
    tree_sol: torch.Tensor | None,
    metrics_sol: torch.Tensor | None,
) -> DiffusionTargets:
    device = graph_inputs.area.device
    block_count = inst.block_count
    if graph_inputs.block_count != block_count:
        raise ValueError("graph_inputs block_count must match instance block_count")

    xywh = _fp_sol_xywh(fp_sol, block_count, device)
    scale = max(float(graph_inputs.scale), 1.0)
    center = torch.stack(
        (xywh[:, 0] + 0.5 * xywh[:, 2], xywh[:, 1] + 0.5 * xywh[:, 3]), dim=1
    ) / scale
    log_aspect = torch.log(xywh[:, 2] / xywh[:, 3].clamp_min(1e-6))
    tree_edges = parse_tree_sol_edges(tree_sol, block_count)
    pair_axis_label, tree_pair_mask, tree_side_label = _pair_axis_labels(
        xywh,
        graph_inputs.pair_index,
        tree_edges,
    )
    metrics = split_metrics_sol(metrics_sol)
    return DiffusionTargets(
        center=center,
        log_aspect=log_aspect,
        pair_axis_label=pair_axis_label,
        tree_pair_mask=tree_pair_mask,
        tree_side_label=tree_side_label,
        instance_stats=metrics.instance_stats.to(device=device, dtype=torch.float32),
        quality_labels=metrics.quality_labels.to(device=device, dtype=torch.float32),
    )
