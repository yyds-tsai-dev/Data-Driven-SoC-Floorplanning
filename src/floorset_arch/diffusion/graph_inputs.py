from __future__ import annotations

import math
from itertools import combinations

import torch

from floorset_arch.diffusion.contracts import (
    DIFFUSION_FEATURE_VERSION,
    DiffusionGraphInputs,
)
from floorset_arch.features import (
    build_anchor_hgt_graph_inputs,
    build_anchor_node_features,
)
from floorset_arch.models import Instance


def _id_tensor(values: dict[int, int], block_count: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros(block_count, dtype=torch.long, device=device)
    for block, value in values.items():
        if 0 <= block < block_count:
            out[block] = int(value)
    return out


def _constraint_ids(inst: Instance, column: int, device: torch.device) -> torch.Tensor:
    if inst.constraints is None or inst.constraints.dim() < 2 or inst.constraints.shape[1] <= column:
        return torch.zeros(inst.block_count, dtype=torch.long, device=device)
    return inst.constraints[: inst.block_count, column].long().to(device)


def _block_mask(blocks: set[int], block_count: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(block_count, dtype=torch.bool, device=device)
    for block in blocks:
        if 0 <= block < block_count:
            mask[block] = True
    return mask


def _pair_set_from_instance(inst: Instance) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for i_f, j_f, _weight_f in inst.valid_b2b.tolist():
        i, j = int(i_f), int(j_f)
        if 0 <= i < inst.block_count and 0 <= j < inst.block_count and i != j:
            pairs.add(tuple(sorted((i, j))))
    for groups in (inst.cluster_groups, inst.mib_groups):
        for blocks in groups.values():
            valid_blocks = sorted({int(block) for block in blocks if 0 <= block < inst.block_count})
            pairs.update(tuple(pair) for pair in combinations(valid_blocks, 2))
    if not pairs:
        pairs.update(combinations(range(inst.block_count), 2))
    return sorted(pairs)


def _b2b_weights(inst: Instance) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = {}
    for i_f, j_f, weight_f in inst.valid_b2b.tolist():
        i, j = int(i_f), int(j_f)
        if 0 <= i < inst.block_count and 0 <= j < inst.block_count and i != j:
            pair = tuple(sorted((i, j)))
            weights[pair] = weights.get(pair, 0.0) + max(float(weight_f), 0.0)
    return weights


def _pair_features(
    inst: Instance,
    pairs: list[tuple[int, int]],
    area: torch.Tensor,
    boundary_codes: torch.Tensor,
    cluster_ids: torch.Tensor,
    mib_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    b2b_weights = _b2b_weights(inst)
    max_b2b = max(b2b_weights.values(), default=1.0)
    rows: list[list[float]] = []
    area_cpu = area.detach().cpu().float().clamp_min(1e-6)
    boundary_cpu = boundary_codes.detach().cpu().long()
    cluster_cpu = cluster_ids.detach().cpu().long()
    mib_cpu = mib_ids.detach().cpu().long()
    for i, j in pairs:
        pair = tuple(sorted((i, j)))
        b2b_weight = b2b_weights.get(pair, 0.0)
        same_cluster = cluster_cpu[i].item() != 0 and cluster_cpu[i].item() == cluster_cpu[j].item()
        same_mib = mib_cpu[i].item() != 0 and mib_cpu[i].item() == mib_cpu[j].item()
        same_boundary = (
            boundary_cpu[i].item() != 0 and boundary_cpu[i].item() == boundary_cpu[j].item()
        )
        rows.append(
            [
                1.0 if b2b_weight > 0.0 else 0.0,
                b2b_weight / max(max_b2b, 1.0),
                1.0 if same_cluster else 0.0,
                1.0 if same_mib else 0.0,
                1.0 if same_boundary else 0.0,
                float(torch.log(area_cpu[i] / area_cpu[j]).item()),
            ]
        )
    if not rows:
        return torch.empty((0, 6), dtype=torch.float32, device=device)
    return torch.tensor(rows, dtype=torch.float32, device=device)


# Near-constant golden whitespace fraction measured across the validation set
# (scripts/probes/outline_leak_probe.py). Declared as a GLOBAL prior, not a
# per-case leak: the estimated canvas area is total_block_area / (1 - ws).
_WHITESPACE_PRIOR = 0.0289


def estimate_frame(inst: Instance, area: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Deterministic canvas (W, H) estimate from solve()-visible inputs only.

    Area-closure gives the magnitude (side = sqrt(total_area / (1 - ws_prior)));
    the pins bounding-box aspect gives the shape; a hard pins-extent floor only
    raises a side when it clearly exceeds the closure estimate. Never touches the
    golden layout, so it is identical between training (target_positions=None)
    and inference. Preplaced extents are deliberately NOT used, to keep the
    estimator symmetric across the two entrypoints.
    """
    total_area = float(area.float().clamp_min(1.0).sum().clamp_min(1.0).item())
    side = math.sqrt(total_area / max(1e-6, 1.0 - _WHITESPACE_PRIOR))
    w_pin = h_pin = 0.0
    if inst.pins_pos.numel() > 0:
        pins = inst.pins_pos.float()
        valid = (pins[:, 0] != -1.0) & (pins[:, 1] != -1.0)
        if bool(valid.any().item()):
            vp = pins[valid]
            w_pin = float(vp[:, 0].clamp_min(0.0).max().item())
            h_pin = float(vp[:, 1].clamp_min(0.0).max().item())
    aspect = (w_pin / h_pin) if (w_pin > 1e-6 and h_pin > 1e-6) else 1.0
    w_area = side * math.sqrt(aspect)
    h_area = side / math.sqrt(aspect)
    w_est = max(w_area, w_pin) if w_pin > w_area * 1.5 else w_area
    h_est = max(h_area, h_pin) if h_pin > h_area * 1.5 else h_area
    return torch.tensor(
        [max(w_est, 1.0), max(h_est, 1.0)], dtype=torch.float32, device=device
    )


def _pin_direction_features(
    inst: Instance,
    frame: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Per-block pin I/O direction features (feature_version >= 2).

    6 dims: frame-relative pin-centroid offset (ox, oy in ~[-1, 1]) plus the
    fraction of pin weight sitting within a 2% band of each of the four canvas
    walls (left, right, bottom, top). Encodes the pins-on-boundary I/O leak
    (~78.7% of weighted pins) as a first-class per-block anchor.
    """
    n = inst.block_count
    width = max(float(frame[0].item()), 1.0)
    height = max(float(frame[1].item()), 1.0)
    eps = 0.02
    pin_cx = torch.zeros(n, device=device)
    pin_cy = torch.zeros(n, device=device)
    pin_w = torch.zeros(n, device=device)
    wall = torch.zeros(n, 4, device=device)
    npins = inst.pins_pos.shape[0]
    pins = inst.pins_pos.float() if npins > 0 else None
    for p_f, b_f, w_f in inst.valid_p2b.tolist():
        p = int(p_f)
        b = int(b_f)
        weight = max(float(w_f), 0.0)
        if not (0 <= b < n and 0 <= p < npins and weight > 0.0):
            continue
        px = float(pins[p, 0])
        py = float(pins[p, 1])
        if px == -1.0 and py == -1.0:
            continue
        pin_cx[b] += weight * px
        pin_cy[b] += weight * py
        pin_w[b] += weight
        if px <= eps * width:
            wall[b, 0] += weight
        if px >= (1.0 - eps) * width:
            wall[b, 1] += weight
        if py <= eps * height:
            wall[b, 2] += weight
        if py >= (1.0 - eps) * height:
            wall[b, 3] += weight
    has = pin_w > 0.0
    denom = pin_w.clamp_min(1e-6)
    cx = torch.where(has, pin_cx / denom, torch.zeros_like(pin_cx))
    cy = torch.where(has, pin_cy / denom, torch.zeros_like(pin_cy))
    ox = torch.where(has, (cx - width / 2.0) / (width / 2.0), torch.zeros_like(cx))
    oy = torch.where(has, (cy - height / 2.0) / (height / 2.0), torch.zeros_like(cy))
    wall_frac = torch.where(
        has.view(-1, 1), wall / denom.view(-1, 1), torch.zeros_like(wall)
    )
    return torch.cat([ox.view(-1, 1), oy.view(-1, 1), wall_frac], dim=1)


def _global_features(
    inst: Instance,
    area: torch.Tensor,
    scale: float,
    pair_count: int,
    fixed_mask: torch.Tensor,
    preplaced_mask: torch.Tensor,
    frame: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Instance-global conditioning (feature_version >= 2).

    Every entry carries per-case variance: three connectivity densities, a pair
    density, fixed/preplaced fractions, absolute scale, and the estimated canvas
    (log W, log H, log-aspect), plus a block-count band. The legacy v1 vector
    had four degenerate entries (block_count/n, areaSum/areaSum, scale/scale,
    and the fixed+preplaced+movable partition) that were all constant 1.0.
    """
    n = max(float(inst.block_count), 1.0)
    w_est = max(float(frame[0].item()), 1.0)
    h_est = max(float(frame[1].item()), 1.0)
    values = [
        float(inst.pins_pos.shape[0]) / n,
        float(inst.valid_b2b.shape[0]) / n,
        float(inst.valid_p2b.shape[0]) / n,
        float(pair_count) / max(n * n, 1.0),
        float(fixed_mask.float().mean().item()) if inst.block_count else 0.0,
        float(preplaced_mask.float().mean().item()) if inst.block_count else 0.0,
        math.log(max(float(scale), 1.0)) / 10.0,
        math.log(w_est) / 10.0,
        math.log(h_est) / 10.0,
        math.log(w_est / h_est),
        n / 120.0,
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)


def build_diffusion_graph_inputs(
    inst: Instance,
    device: torch.device | None = None,
) -> DiffusionGraphInputs:
    if inst.block_count <= 0:
        raise ValueError("build_diffusion_graph_inputs requires positive block_count")
    device = device or inst.area_targets.device
    block_features, scale = build_anchor_node_features(inst, device=device)
    hgt_graph = build_anchor_hgt_graph_inputs(
        inst,
        device=device,
        block_features=block_features,
        scale=scale,
    )
    node_features = hgt_graph.node_features
    edge_index = hgt_graph.edge_index
    edge_attr = hgt_graph.edge_attr
    area = inst.area_targets[: inst.block_count].float().to(device)
    frame = estimate_frame(inst, area, device)
    # feature_version >= 2: append per-block pin I/O direction features to the
    # direct-path raw block features. The shared HGT node features (block_features)
    # are left at their original dim so v5 anchor semantics are untouched.
    pin_dir = _pin_direction_features(inst, frame, device)
    raw_block_features = torch.cat([block_features, pin_dir], dim=1)
    fixed_mask = _block_mask(inst.fixed, inst.block_count, device)
    preplaced_mask = _block_mask(inst.preplaced, inst.block_count, device)
    movable_mask = ~(fixed_mask | preplaced_mask)
    boundary_codes = _id_tensor(inst.boundary, inst.block_count, device)
    cluster_ids = _constraint_ids(inst, 3, device)
    mib_ids = _constraint_ids(inst, 2, device)

    pairs = _pair_set_from_instance(inst)
    pair_index = (
        torch.tensor(pairs, dtype=torch.long, device=device)
        if pairs
        else torch.empty((0, 2), dtype=torch.long, device=device)
    )
    raw_pair_features = _pair_features(
        inst,
        pairs,
        area,
        boundary_codes,
        cluster_ids,
        mib_ids,
        device,
    )

    return DiffusionGraphInputs(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        relation_specs=hgt_graph.relation_specs,
        raw_block_features=raw_block_features,
        raw_pair_features=raw_pair_features,
        pair_index=pair_index,
        area=area,
        scale=float(scale),
        fixed_mask=fixed_mask,
        preplaced_mask=preplaced_mask,
        movable_mask=movable_mask,
        boundary_codes=boundary_codes,
        cluster_ids=cluster_ids,
        mib_ids=mib_ids,
        global_features=_global_features(
            inst,
            area,
            float(scale),
            len(pairs),
            fixed_mask,
            preplaced_mask,
            frame,
            device,
        ),
        frame=frame,
        feature_version=DIFFUSION_FEATURE_VERSION,
    )
