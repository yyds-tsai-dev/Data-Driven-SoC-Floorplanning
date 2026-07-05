from __future__ import annotations

import torch
import torch.nn.functional as F

from floorset_arch.models import Placement, Rect
from floorset_arch.parser import parse_instance
from floorset_arch.repair import soft_violation_counts


def _placement_from_fp_sol(fp_sol: torch.Tensor, block_count: int) -> Placement:
    gt = fp_sol[:block_count].detach().cpu().float()
    rects = {}
    for block in range(block_count):
        width, height, x, y = [float(value) for value in gt[block].tolist()]
        rects[block] = Rect(x, y, width, height)
    return Placement(rects)


def fp_sol_soft_violations(
    fp_sol: torch.Tensor,
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
) -> tuple[int, int, int]:
    block_count = int((area_targets.detach().flatten() != -1).sum().item())
    inst = parse_instance(
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        None,
    )
    return soft_violation_counts(inst, _placement_from_fp_sol(fp_sol, block_count))


def is_constraint_clean_training_sample(
    fp_sol: torch.Tensor,
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
) -> bool:
    return sum(
        fp_sol_soft_violations(
            fp_sol,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
        )
    ) == 0


def build_anchor_targets(
    fp_sol: torch.Tensor, block_count: int, scale: float, device: torch.device
) -> dict[str, torch.Tensor]:
    gt = fp_sol[:block_count].float().to(device)
    width = gt[:, 0].clamp_min(1e-6)
    height = gt[:, 1].clamp_min(1e-6)
    x = gt[:, 2]
    y = gt[:, 3]
    anchor = torch.stack([x + width / 2.0, y + height / 2.0], dim=1) / max(
        float(scale), 1.0
    )
    log_aspect = torch.log(width / height).clamp(-2.5, 2.5)
    center_sum = anchor.sum(dim=1)
    span = (center_sum.max() - center_sum.min()).clamp_min(1e-6)
    priority = 1.0 - (center_sum - center_sum.min()) / span
    return {"anchor": anchor, "log_aspect": log_aspect, "priority": priority}


def constraint_weights(
    constraints: torch.Tensor, block_count: int, device: torch.device, args
) -> torch.Tensor:
    weights = torch.ones(block_count, device=device)
    if constraints is None or constraints.dim() <= 1:
        return weights
    c = constraints[:block_count].to(device)
    if c.shape[1] > 0:
        weights = weights + 0.10 * (c[:, 0] != 0).float()
    if c.shape[1] > 1:
        weights = weights + 0.10 * (c[:, 1] != 0).float()
    if c.shape[1] > 2:
        weights = weights + args.mib_weight_boost * (c[:, 2] != 0).float()
    if c.shape[1] > 3:
        weights = weights + args.cluster_weight_boost * (c[:, 3] != 0).float()
    if c.shape[1] > 4:
        weights = weights + args.boundary_weight_boost * (c[:, 4] != 0).float()
    return weights


def weighted_smooth_l1(
    pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if loss.dim() > 1:
        loss = loss.sum(dim=1)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def sample_pairs(block_count: int, max_pairs: int, device: torch.device) -> torch.Tensor:
    pairs = [(i, j) for i in range(block_count) for j in range(i + 1, block_count)]
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    pair_tensor = torch.tensor(pairs, dtype=torch.long, device=device)
    if max_pairs > 0 and pair_tensor.shape[0] > max_pairs:
        pair_tensor = pair_tensor[
            torch.randperm(pair_tensor.shape[0], device=device)[:max_pairs]
        ]
    return pair_tensor


def order_aux_loss(
    pred_anchor: torch.Tensor, target_anchor: torch.Tensor, args
) -> tuple[torch.Tensor, float, float]:
    device = pred_anchor.device
    pairs = sample_pairs(pred_anchor.shape[0], args.order_pairs, device)
    if pairs.numel() == 0:
        return pred_anchor.sum() * 0.0, 0.0, 1.0
    i = pairs[:, 0]
    j = pairs[:, 1]
    gt_delta = target_anchor[j] - target_anchor[i]
    abs_dx = gt_delta[:, 0].abs()
    abs_dy = gt_delta[:, 1].abs()
    clear_x = (abs_dx > args.clear_ratio * abs_dy) & (abs_dx > args.min_order_gap)
    clear_y = (abs_dy > args.clear_ratio * abs_dx) & (abs_dy > args.min_order_gap)

    loss = pred_anchor.sum() * 0.0
    used = 0
    acc_sum = 0.0
    acc_count = 0
    if clear_x.any():
        sign = torch.sign(gt_delta[clear_x, 0])
        pred_delta = pred_anchor[j[clear_x], 0] - pred_anchor[i[clear_x], 0]
        loss = loss + F.softplus(-sign * pred_delta / args.order_temp).mean()
        used += int(clear_x.sum().item())
        acc_sum += float(((pred_delta * sign) > 0).float().mean().item())
        acc_count += 1
    if clear_y.any():
        sign = torch.sign(gt_delta[clear_y, 1])
        pred_delta = pred_anchor[j[clear_y], 1] - pred_anchor[i[clear_y], 1]
        loss = loss + F.softplus(-sign * pred_delta / args.order_temp).mean()
        used += int(clear_y.sum().item())
        acc_sum += float(((pred_delta * sign) > 0).float().mean().item())
        acc_count += 1
    return loss, used / max(float(pairs.shape[0]), 1.0), acc_sum / max(acc_count, 1)


def build_pairwise_relation_targets(
    fp_sol: torch.Tensor,
    pairs: torch.Tensor,
    min_gap: float,
    clear_ratio: float,
) -> dict[str, torch.Tensor]:
    device = fp_sol.device
    if pairs.numel() == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        return {"x_label": empty, "y_label": empty, "mask": empty.bool()}
    gt = fp_sol.float()
    centers = torch.stack([gt[:, 2] + gt[:, 0] / 2.0, gt[:, 3] + gt[:, 1] / 2.0], dim=1)
    i = pairs[:, 0].to(device)
    j = pairs[:, 1].to(device)
    delta = centers[j] - centers[i]
    abs_dx = delta[:, 0].abs()
    abs_dy = delta[:, 1].abs()
    clear_x = (abs_dx > clear_ratio * abs_dy) & (abs_dx > min_gap)
    clear_y = (abs_dy > clear_ratio * abs_dx) & (abs_dy > min_gap)
    x_label = (delta[:, 0] > 0).float() * clear_x.float()
    y_label = (delta[:, 1] > 0).float() * clear_y.float()
    return {"x_label": x_label, "y_label": y_label, "mask": clear_x | clear_y}


def pairwise_relation_loss(
    pair_logits: torch.Tensor,
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, float]:
    mask = targets["mask"].to(pair_logits.device)
    if mask.numel() == 0 or not mask.any():
        return pair_logits.sum() * 0.0, 1.0
    labels = torch.stack(
        [targets["x_label"].to(pair_logits.device), targets["y_label"].to(pair_logits.device)],
        dim=1,
    )
    loss = F.binary_cross_entropy_with_logits(pair_logits[mask], labels[mask])
    pred = (pair_logits[mask] > 0).float()
    acc = float((pred == labels[mask]).float().mean().item())
    return loss, acc


# Decoder class order (must match gen_decoder_probe.build_order_dags):
#   0 = i left-of j   (x-axis, dx > 0)
#   1 = j left-of i   (x-axis, dx < 0)
#   2 = i below j     (y-axis, dy > 0)
#   3 = j below i     (y-axis, dy < 0)
PAIR_AXIS4_CLASSES = 4


def build_order_axis4_targets(
    fp_sol: torch.Tensor,
    pairs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Composite 4-class order label per pair (i, j), i<j, matching the
    order-faithful decoder's ``build_order_dags`` rule EXACTLY:

      * centroids from the (golden) hint layout,
      * separating axis = the one with the LARGER NORMALIZED centroid
        separation ``|d_axis| / (half_i + half_j)`` on that axis,
      * tie (sep_x >= sep_y) resolves to the x-axis,
      * direction = sign of the centroid delta on the chosen axis, with the
        ``d==0`` degenerate case resolved to ``i-before-j`` (i<j always here).

    ``fp_sol`` rows are (width, height, x, y) as elsewhere in this module.
    Returns per-pair integer class labels and a near-tie weight in [1, 2] that
    up-weights pairs where the two normalized separations are close (the hard
    cases the geometric extractor still resolves but a weak head fumbles).
    Every pair is labelled — there is no clear-ratio mask.
    """
    device = fp_sol.device
    if pairs.numel() == 0:
        empty_l = torch.empty((0,), dtype=torch.long, device=device)
        empty_f = torch.empty((0,), dtype=torch.float32, device=device)
        return {"label": empty_l, "tie_weight": empty_f}
    gt = fp_sol.float()
    width = gt[:, 0]
    height = gt[:, 1]
    cx = gt[:, 2] + width / 2.0
    cy = gt[:, 3] + height / 2.0
    hx = width / 2.0
    hy = height / 2.0
    i = pairs[:, 0].to(device)
    j = pairs[:, 1].to(device)
    dx = cx[j] - cx[i]
    dy = cy[j] - cy[i]
    sum_hx = (hx[i] + hx[j]).clamp_min(0.0)
    sum_hy = (hy[i] + hy[j]).clamp_min(0.0)
    sep_x = torch.where(sum_hx > 0, dx.abs() / sum_hx, torch.zeros_like(dx))
    sep_y = torch.where(sum_hy > 0, dy.abs() / sum_hy, torch.zeros_like(dy))
    use_x = sep_x >= sep_y  # tie -> x-axis, matching the decoder
    # Direction: dx>0 (or dx==0 with i<j) -> i before j on x -> class 0 else 1.
    x_i_first = (dx > 0) | (dx == 0)
    y_i_first = (dy > 0) | (dy == 0)
    label = torch.where(
        use_x,
        torch.where(x_i_first, torch.zeros_like(i), torch.ones_like(i)),
        torch.where(y_i_first, torch.full_like(i, 2), torch.full_like(i, 3)),
    ).long()
    # Near-tie weight: 1 when the losing axis has ~0 separation, up to 2 when the
    # two normalized separations are equal (max ambiguity).
    max_sep = torch.maximum(sep_x, sep_y).clamp_min(1e-9)
    min_sep = torch.minimum(sep_x, sep_y)
    tie_weight = 1.0 + (min_sep / max_sep).clamp(0.0, 1.0)
    return {"label": label, "tie_weight": tie_weight}


def pair_edge_weights(
    pairs: torch.Tensor,
    valid_b2b: torch.Tensor,
    block_count: int,
    alpha: float,
) -> torch.Tensor:
    """Per-pair multiplier ``1 + alpha * norm_netweight`` where norm_netweight is
    the (log1p, mean-normalized) b2b connectivity weight between the pair's two
    blocks, or 0 when they share no net. Returns ones when alpha<=0."""
    device = pairs.device
    if pairs.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)
    if alpha <= 0.0 or valid_b2b is None or valid_b2b.numel() == 0:
        return torch.ones((pairs.shape[0],), dtype=torch.float32, device=device)
    net: dict[tuple[int, int], float] = {}
    for i_f, j_f, w_f in valid_b2b.tolist():
        i, j, w = int(i_f), int(j_f), max(float(w_f), 0.0)
        if 0 <= i < block_count and 0 <= j < block_count and i != j:
            key = (i, j) if i < j else (j, i)
            net[key] = net.get(key, 0.0) + w
    if net:
        vals = torch.tensor(list(net.values()), dtype=torch.float32)
        log_mean = torch.log1p(vals).mean().clamp_min(1e-6)
    else:
        log_mean = torch.tensor(1.0)
    out = torch.ones((pairs.shape[0],), dtype=torch.float32, device=device)
    for idx, (i_f, j_f) in enumerate(pairs.tolist()):
        i, j = int(i_f), int(j_f)
        key = (i, j) if i < j else (j, i)
        w = net.get(key)
        if w is not None:
            out[idx] = 1.0 + alpha * float(
                (torch.log1p(torch.tensor(w)) / log_mean).item()
            )
    return out


def order_axis4_loss(
    pair_axis4_logits: torch.Tensor,
    targets: dict[str, torch.Tensor],
    class_weight: torch.Tensor | None = None,
    edge_weight: torch.Tensor | None = None,
    tie_alpha: float = 0.0,
) -> tuple[torch.Tensor, float, float]:
    """Class-weighted cross-entropy over the 4 decoder classes on ALL sampled
    pairs. Returns (loss, composite-accuracy, axis-accuracy). Accuracies are the
    consumer-facing metrics: composite = axis+direction agreement with the
    decoder label; axis = agreement on x-vs-y only.
    """
    label = targets["label"].to(pair_axis4_logits.device)
    if label.numel() == 0:
        return pair_axis4_logits.sum() * 0.0, 1.0, 1.0
    per = F.cross_entropy(
        pair_axis4_logits,
        label,
        weight=class_weight.to(pair_axis4_logits.device) if class_weight is not None else None,
        reduction="none",
    )
    weight = torch.ones_like(per)
    if tie_alpha > 0.0 and "tie_weight" in targets:
        weight = weight * (
            1.0 + tie_alpha * (targets["tie_weight"].to(per.device) - 1.0)
        )
    if edge_weight is not None:
        weight = weight * edge_weight.to(per.device)
    loss = (per * weight).sum() / weight.sum().clamp_min(1e-6)
    pred = pair_axis4_logits.argmax(dim=1)
    composite_acc = float((pred == label).float().mean().item())
    pred_axis = (pred >= 2).long()
    true_axis = (label >= 2).long()
    axis_acc = float((pred_axis == true_axis).float().mean().item())
    return loss, composite_acc, axis_acc


def edge_delta_loss(
    pred_anchor: torch.Tensor, target_anchor: torch.Tensor, valid_b2b: torch.Tensor
) -> torch.Tensor:
    edges = []
    weights = []
    for i_f, j_f, weight_f in valid_b2b.tolist():
        i = int(i_f)
        j = int(j_f)
        weight = max(float(weight_f), 0.0)
        if 0 <= i < pred_anchor.shape[0] and 0 <= j < pred_anchor.shape[0] and i != j:
            edges.append((i, j))
            weights.append(weight)
    if not edges:
        return pred_anchor.sum() * 0.0
    edge_t = torch.tensor(edges, dtype=torch.long, device=pred_anchor.device)
    weight_t = torch.log1p(
        torch.tensor(weights, dtype=torch.float32, device=pred_anchor.device)
    )
    weight_t = weight_t / weight_t.mean().clamp_min(1.0)
    pred_delta = pred_anchor[edge_t[:, 1]] - pred_anchor[edge_t[:, 0]]
    target_delta = target_anchor[edge_t[:, 1]] - target_anchor[edge_t[:, 0]]
    per = F.smooth_l1_loss(pred_delta, target_delta, reduction="none").sum(dim=1)
    return (per * weight_t).mean()


def compute_anchor_losses(
    pred: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: torch.Tensor,
    valid_b2b: torch.Tensor,
    args,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    anchor = weighted_smooth_l1(pred["anchor"], targets["anchor"], weights)
    aspect = weighted_smooth_l1(pred["log_aspect"], targets["log_aspect"], weights)
    priority = weighted_smooth_l1(pred["priority"], targets["priority"], weights)
    order, ord_frac, ord_acc = order_aux_loss(pred["anchor"], targets["anchor"], args)
    edge = edge_delta_loss(pred["anchor"], targets["anchor"], valid_b2b)
    loss = (
        args.anchor_weight * anchor
        + args.aspect_weight * aspect
        + args.priority_weight * priority
        + args.order_weight * order
        + args.edge_weight * edge
    )
    return loss, {
        "anchor": anchor,
        "aspect": aspect,
        "priority": priority,
        "order": order,
        "edge": edge,
        "ord_frac": ord_frac,
        "ord_acc": ord_acc,
    }
