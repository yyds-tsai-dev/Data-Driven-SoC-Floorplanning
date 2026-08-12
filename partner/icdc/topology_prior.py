"""Leak-free sparse topology labels and differentiable topology losses."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import torch

from . import tfdl as T
from . import engine
from .topology_data import ContactLabel, SparseEdge, SparseTopologyBatch, TopologyLabel


@dataclass(frozen=True)
class ProposalConfig:
    axis_exchange_cap: int = 8
    pin_repair_cap: int = 8
    group_contact_cap: int = 8
    total_cap: int = 32

    def __post_init__(self):
        for value in (self.axis_exchange_cap, self.pin_repair_cap, self.group_contact_cap, self.total_cap):
            if type(value) is not int or value < 0:
                raise ValueError("proposal caps must be non-negative integers")


@dataclass(frozen=True)
class ProposalResult:
    name: str
    rects: torch.Tensor
    legal: torch.Tensor
    drift: torch.Tensor
    hard_checks: Dict[str, bool]
    cost: Optional[float] = None
    label: Optional[TopologyLabel] = None


def is_acyclic(n: int, edges: Sequence[Tuple[int, int]]) -> bool:
    if type(n) is not int or n <= 0 or not isinstance(edges, (list, tuple)):
        return False
    graph = [[] for _ in range(n)]
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2 or any(type(x) is not int for x in edge):
            return False
        a, b = edge
        if not (0 <= a < n and 0 <= b < n) or a == b:
            return False
        graph[a].append(b)
    state = [0] * n
    def visit(node: int) -> bool:
        if state[node] == 1:
            return False
        if state[node] == 2:
            return True
        state[node] = 1
        if any(not visit(child) for child in graph[node]):
            return False
        state[node] = 2
        return True
    return all(visit(i) for i in range(n))


def _rect_cpu(rects: torch.Tensor, n: int, positive_sizes: bool = False) -> bool:
    if not isinstance(rects, torch.Tensor) or rects.device.type != "cpu":
        return False
    if not rects.is_floating_point() or rects.ndim not in (2, 3) or rects.shape[-1] != 4:
        return False
    shape_ok = rects.shape == (n, 4) if rects.ndim == 2 else rects.shape == (1, n, 4)
    if not shape_ok or not bool(torch.isfinite(rects).all()):
        return False
    return not positive_sizes or bool((rects[..., 2:] > 0).all())


def _case_tp(case: Mapping[str, Any], n: int) -> Optional[torch.Tensor]:
    value = case.get("tp")
    try:
        tp = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    except Exception:
        return None
    if tp.shape != (n, 4) or not bool(torch.isfinite(tp).all()):
        return None
    return tp


def matches_preplaced_origins(rects: torch.Tensor, case: Mapping[str, Any]) -> bool:
    try:
        if not isinstance(case, Mapping):
            return False
        n = case.get("n")
        if type(n) is not int or n <= 0:
            return False
        cons = _constraints(case, n)
        tp = _case_tp(case, n)
        if tp is None or not _rect_cpu(rects, n):
            return False
        value = rects[0] if rects.ndim == 3 else rects
        for i, row in enumerate(cons):
            authorized = row[1] != 0 and bool((tp[i, :2] >= 0).all())
            if authorized and not torch.equal(value[i, :2], tp[i, :2]):
                return False
        return True
    except Exception:
        return False


def has_exact_positive_contact(rects: torch.Tensor, a: int, b: int, axis: int,
                               a_before_b: bool, perp_margin: float) -> bool:
    try:
        if type(a) is not int or type(b) is not int or type(axis) is not int:
            return False
        if type(a_before_b) is not bool or axis not in (0, 1) or a == b:
            return False
        if isinstance(perp_margin, bool) or not isinstance(perp_margin, (int, float)):
            return False
        if not math.isfinite(float(perp_margin)) or perp_margin <= 0:
            return False
        n = rects.shape[-2] if isinstance(rects, torch.Tensor) and rects.ndim in (2, 3) else -1
        if not _rect_cpu(rects, n, positive_sizes=True):
            return False
        r = rects[0] if rects.ndim == 3 else rects
        if not (0 <= a < n and 0 <= b < n):
            return False
        perp = 1 - axis
        a_end = r[a, axis] + r[a, axis + 2]
        b_end = r[b, axis] + r[b, axis + 2]
        exact_face = (a_end == r[b, axis]) if a_before_b else (b_end == r[a, axis])
        if not bool(exact_face):
            return False
        overlap = min(r[a, perp] + r[a, perp + 2], r[b, perp] + r[b, perp + 2]) - max(r[a, perp], r[b, perp])
        return bool(overlap >= float(perp_margin) and overlap > 0)
    except Exception:
        return False


def _validate_proposal_case(case: Mapping[str, Any]) -> Tuple[int, torch.Tensor, list[list[int]], torch.Tensor]:
    if not isinstance(case, Mapping):
        raise TypeError("case must be a mapping")
    n = case.get("n")
    if type(n) is not int or n <= 0:
        raise ValueError("n")
    cons = _constraints(case, n)
    area_value = case.get("area")
    if not isinstance(area_value, (list, tuple)) or len(area_value) != n:
        raise ValueError("area")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) or float(value) <= 0
           for value in area_value):
        raise ValueError("area")
    area = torch.tensor(area_value, dtype=torch.float64, device="cpu")
    tp = _case_tp(case, n)
    if tp is None:
        raise ValueError("tp")
    for index, row in enumerate(cons):
        fixed_or_preplaced = row[0] != 0 or row[1] != 0
        if fixed_or_preplaced and (tp[index, 2] <= 0 or tp[index, 3] <= 0):
            raise ValueError("fixed dimensions")
        if row[1] != 0 and (tp[index, 0] < 0 or tp[index, 1] < 0):
            raise ValueError("preplaced origin")
    return n, area, cons, tp


def _authorized_preplaced(cons: Sequence[Sequence[int]], tp: torch.Tensor) -> Tuple[int, ...]:
    return tuple(index for index, row in enumerate(cons)
                 if row[1] != 0 and bool((tp[index, :2] >= 0).all()))


def _pair_gap(rects: torch.Tensor, first: int, second: int, axis: int) -> float:
    a = rects[first]
    b = rects[second]
    return max(float(a[axis] - b[axis] - b[axis + 2]),
               float(b[axis] - a[axis] - a[axis + 2]))


def _pair_state(rects: torch.Tensor, first: int, second: int) -> Tuple[int, int]:
    gap_x = _pair_gap(rects, first, second, 0)
    gap_y = _pair_gap(rects, first, second, 1)
    axis = 0 if gap_x >= gap_y else 1
    first_center = float(rects[first, axis] + rects[first, axis + 2] / 2)
    second_center = float(rects[second, axis] + rects[second, axis + 2] / 2)
    return axis, int((first_center, first) <= (second_center, second))


def _contact_relation(rects: torch.Tensor, first: int, second: int,
                      axis: int, order: int) -> bool:
    if rects.ndim == 3:
        rects = rects[0]
    if order == 1:
        exact = rects[first, axis] + rects[first, axis + 2] == rects[second, axis]
    else:
        exact = rects[second, axis] + rects[second, axis + 2] == rects[first, axis]
    perp = 1 - axis
    overlap = min(float(rects[first, perp] + rects[first, perp + 2]),
                  float(rects[second, perp] + rects[second, perp + 2])) - max(
                      float(rects[first, perp]), float(rects[second, perp]))
    return bool(exact and overlap > 0)


def _cluster_contacts(rects: torch.Tensor, cons: Sequence[Sequence[int]]) -> Tuple[Tuple[int, int, int, int], ...]:
    contacts = []
    for first in range(len(cons)):
        for second in range(first + 1, len(cons)):
            if len(cons[first]) < 4 or len(cons[second]) < 4:
                continue
            if cons[first][3] <= 0 or cons[first][3] != cons[second][3]:
                continue
            for axis in (0, 1):
                for order in (0, 1):
                    if _contact_relation(rects, first, second, axis, order):
                        contacts.append((cons[first][3], first, second, axis, order))
    return tuple(contacts)


def _proposal_fingerprint(rects: torch.Tensor, cons: Sequence[Sequence[int]]) -> Tuple[Tuple[Any, ...], ...]:
    fingerprint = []
    for first in range(rects.shape[0]):
        for second in range(first + 1, rects.shape[0]):
            axis, order = _pair_state(rects, first, second)
            fingerprint.append(("pair", first, second, axis, order))
    fingerprint.extend(("contact", gid, first, second, axis, order)
                       for gid, first, second, axis, order in _cluster_contacts(rects, cons))
    return tuple(fingerprint)


def _pin_mask(cons: Sequence[Sequence[int]], tp: torch.Tensor) -> Tuple[bool, ...]:
    return tuple(index in _authorized_preplaced(cons, tp) for index in range(len(cons)))


def _place_for_pair(rects: torch.Tensor, first: int, second: int,
                    axis: int, order: int, moved: int) -> torch.Tensor:
    stationary = second if moved == first else first
    m = rects[moved]
    s = rects[stationary]
    perp = 1 - axis
    before = (moved == first) == bool(order)
    sigma = -1.0 if before else 1.0
    cm = [float(m[d] + m[d + 2] / 2) for d in (0, 1)]
    cs = [float(s[d] + s[d + 2] / 2) for d in (0, 1)]
    z0a = cm[axis] - cs[axis]
    v0 = cm[perp] - cs[perp]
    ha = (float(m[axis + 2]) + float(s[axis + 2])) / 2
    hp = (float(m[perp + 2]) + float(s[perp + 2])) / 2
    delta = ha - hp
    u0 = sigma * z0a
    projected = []
    if u0 >= 0 and u0 >= abs(v0) + delta:
        projected.append((u0, v0))
    if delta <= 0:
        projected.append((0.0, min(max(v0, delta), -delta)))
    vr = max(0.0, -delta, (u0 + v0 - delta) / 2)
    projected.append((vr + delta, vr))
    vl = min(0.0, delta, (delta - u0 + v0) / 2)
    projected.append((delta - vl, vl))

    # Include the unmodified placement independently: it is the exact nearest
    # answer whenever it already realizes the requested classifier state.
    targets = [(u0, v0)]
    targets.extend(projected)
    best = None
    best_dist = None
    target_centers = []
    for u, v in targets:
        target_centers.append((cs[axis] + sigma * u, cs[perp] + v))
    # Bounded nextafter ladders recover strict floating-point classifier states
    # while validating every actual rectangle state.
    ladder_bases = list(target_centers)
    for ca, cp in ladder_bases:
        for ka in range(17):
            aa = ca
            for _ in range(ka):
                aa = math.nextafter(aa, math.inf if sigma > 0 else -math.inf)
            for kp in range(17):
                pp = cp
                for _ in range(kp):
                    pp = math.nextafter(pp, cs[perp])
                trial = rects.clone()
                trial[moved, axis] = aa - float(m[axis + 2]) / 2
                trial[moved, perp] = pp - float(m[perp + 2]) / 2
                if _pair_state(trial, first, second) != (axis, order):
                    continue
                dist = sum((float(trial[moved, d] - m[d])) ** 2 for d in (0, 1))
                if best_dist is None or dist < best_dist:
                    best, best_dist = trial, dist
    return best if best is not None else rects.clone()


def _repair_preplaced(rects: torch.Tensor, cons: Sequence[Sequence[int]],
                      tp: torch.Tensor) -> torch.Tensor:
    repaired = rects.clone()
    for index in _authorized_preplaced(cons, tp):
        repaired[index, :2] = tp[index, :2]
    return repaired


def _bool_result(value: Any) -> bool:
    value_type = type(value)
    return value_type is bool or (value_type.__module__ == "numpy" and value_type.__name__ == "bool")


def _admission_run(seed: torch.Tensor, mask: torch.Tensor, pinned: torch.Tensor,
                   pin_xy: torch.Tensor, boundary: torch.Tensor, case: Mapping[str, Any],
                   exact: bool) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    try:
        legal, drift = T.tfdl(seed.clone(), mask, pinned, pin_xy=pin_xy,
                              boundary_code=boundary, exact=exact)
    except Exception:
        return None
    if (not isinstance(legal, torch.Tensor) or not isinstance(drift, torch.Tensor)
            or legal.shape != seed.shape or drift.shape != (1, seed.shape[1], 2)
            or legal.dtype is not torch.float64 or drift.dtype is not torch.float64
            or legal.device.type != "cpu" or drift.device.type != "cpu"
            or not bool(torch.isfinite(legal).all()) or not bool(torch.isfinite(drift).all())
            or torch.count_nonzero(drift).item() != 0
            or not matches_preplaced_origins(legal, case)):
        return None
    return legal, drift


def pin_feasible_then_exact_tfdl(
    proposal: torch.Tensor, case: Mapping[str, Any]
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    try:
        n, area, cons, tp = _validate_proposal_case(case)
        if (not isinstance(proposal, torch.Tensor) or proposal.device.type != "cpu"
                or proposal.ndim != 2 or proposal.dtype is not torch.float64
                or not _rect_cpu(proposal, n, True)):
            return None
        if not matches_preplaced_origins(proposal, case):
            return None
        seed = proposal.clone().unsqueeze(0)
        cons_tensor = torch.tensor(cons, dtype=torch.long, device="cpu").unsqueeze(0)
        area_tensor = area.unsqueeze(0)
        mask = area_tensor > 0
        authorized = (tp[:, 0] >= 0) & (tp[:, 1] >= 0)
        pinned = (cons_tensor[:, :, 1] != 0) & authorized.unsqueeze(0) & mask
        pin_xy = tp[:, :2].unsqueeze(0)
        boundary = torch.tensor(
            [[row[4] if len(row) == 5 else 0 for row in cons]],
            dtype=torch.long, device="cpu")
        if _admission_run(seed, mask, pinned, pin_xy, boundary, case, False) is None:
            return None
        exact_result = _admission_run(seed, mask, pinned, pin_xy, boundary, case, True)
        if exact_result is None:
            return None
        legal, drift = exact_result
        checks = engine.verify_hard_legal(
            legal.squeeze(0).numpy(), area.numpy(), cons_tensor.squeeze(0).numpy(), tp.numpy())
        if (not isinstance(checks, Mapping) or not checks
                or not all(_bool_result(value) for value in checks.values())
                or not all(bool(value) for value in checks.values())):
            return None
        return legal.squeeze(0), drift.squeeze(0)
    except Exception:
        return None


def _emit_candidate(name: str, candidate: torch.Tensor, kind: str,
                    cap: int, seen: set, names: set, total: int,
                    total_cap: int, cons: Sequence[Sequence[int]]) -> Optional[Tuple[str, torch.Tensor]]:
    current_count = sum(existing.startswith(kind + ":") for existing in names)
    if (total >= total_cap or current_count >= cap or name in names
            or cap <= 0):
        return None
    fingerprint = _proposal_fingerprint(candidate, cons)
    if fingerprint in seen:
        return None
    seen.add(fingerprint)
    names.add(name)
    return name, candidate


def _contact_components(rects: torch.Tensor, cons: Sequence[Sequence[int]]) -> Dict[int, Tuple[Tuple[int, ...], ...]]:
    groups: Dict[int, list[int]] = {}
    for index, row in enumerate(cons):
        if len(row) >= 4 and row[3] > 0:
            groups.setdefault(row[3], []).append(index)
    output: Dict[int, Tuple[Tuple[int, ...], ...]] = {}
    contact_pairs = {(first, second) for _, first, second, _, _ in _cluster_contacts(rects, cons)}
    for gid, members in sorted(groups.items()):
        parent = {member: member for member in members}

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for first in members:
            for second in members:
                if first < second and (first, second) in contact_pairs:
                    parent[find(first)] = find(second)
        components: Dict[int, list[int]] = {}
        for member in members:
            components.setdefault(find(member), []).append(member)
        output[gid] = tuple(sorted((tuple(sorted(component)) for component in components.values())))
    return output


def _contact_candidate(rects: torch.Tensor, first: int, second: int,
                       axis: int, order: int, moved: int) -> torch.Tensor:
    candidate = rects.clone()
    stationary = second if moved == first else first
    stationary_start = float(rects[stationary, axis])
    stationary_end = stationary_start + float(rects[stationary, axis + 2])
    moved_size = float(rects[moved, axis + 2])
    if moved == first:
        before = order == 1
    else:
        before = order == 0
    candidate[moved, axis] = stationary_start - moved_size if before else stationary_end
    candidate[moved, 1 - axis] = rects[stationary, 1 - axis]
    return candidate


def generate_proposals(
    raw_rects: torch.Tensor, case: Mapping[str, Any],
    cfg: ProposalConfig = ProposalConfig()
) -> Iterator[Tuple[str, torch.Tensor]]:
    if type(cfg) is not ProposalConfig:
        raise TypeError("cfg")
    n, _area, cons, tp = _validate_proposal_case(case)
    if (not isinstance(raw_rects, torch.Tensor) or raw_rects.device.type != "cpu"
            or raw_rects.ndim != 2 or not raw_rects.is_floating_point()
            or not _rect_cpu(raw_rects, n, True)):
        raise ValueError("raw_rects")
    base = raw_rects.to(dtype=torch.float64).clone()
    for index, row in enumerate(cons):
        if row[0] != 0 or row[1] != 0:
            base[index, 2:] = tp[index, 2:]
    if cfg.total_cap == 0:
        return
    seen = {_proposal_fingerprint(base, cons)}
    names = {"base"}
    total = 0
    yield "base", base.clone()
    total += 1
    pinned = _pin_mask(cons, tp)

    for first in range(n):
        for second in range(first + 1, n):
            if total >= cfg.total_cap:
                return
            if cfg.axis_exchange_cap <= sum(name.startswith("axis:") for name in names):
                break
            if pinned[first] and pinned[second]:
                continue
            moved = first if pinned[second] else second
            if not pinned[first] and not pinned[second]:
                moved = second
            current_axis, current_order = _pair_state(base, first, second)
            intents = tuple(sorted(((current_axis, 1 - current_order),
                                    (1 - current_axis, 1 - current_order))))
            for axis, order in intents:
                candidate = _place_for_pair(base, first, second, axis, order, moved)
                actual_axis, actual_order = _pair_state(candidate, first, second)
                if (actual_axis, actual_order) != (axis, order):
                    continue
                name = f"axis:{first}:{second}:{actual_axis}:{actual_order}"
                emitted = _emit_candidate(name, candidate, "axis", cfg.axis_exchange_cap,
                                           seen, names, total, cfg.total_cap, cons)
                if emitted is not None:
                    yield emitted
                    total += 1
                    if total >= cfg.total_cap:
                        return

    pin_base = _repair_preplaced(base, cons, tp)
    preplaced = _authorized_preplaced(cons, tp)
    if preplaced:
        for target in preplaced:
            for peer in range(n):
                if peer == target or pinned[peer]:
                    continue
                current_axis, current_order = _pair_state(pin_base, target, peer)
                candidate = _place_for_pair(pin_base, target, peer, current_axis,
                                            1 - current_order, peer)
                actual_axis, actual_order = _pair_state(candidate, target, peer)
                if (actual_axis, actual_order) != (current_axis, 1 - current_order):
                    continue
                if not matches_preplaced_origins(candidate, case):
                    continue
                name = f"pin:{target}:{peer}:{actual_axis}:{actual_order}"
                emitted = _emit_candidate(name, candidate, "pin", cfg.pin_repair_cap,
                                           seen, names, total, cfg.total_cap, cons)
                if emitted is not None:
                    yield emitted
                    total += 1
                    if total >= cfg.total_cap:
                        return
                if sum(name.startswith("pin:") for name in names) >= cfg.pin_repair_cap:
                    break
            if sum(name.startswith("pin:") for name in names) >= cfg.pin_repair_cap:
                break

    contact_base = pin_base
    contact_components = _contact_components(contact_base, cons)
    for gid, components in sorted(contact_components.items()):
        for left_index, left in enumerate(components):
            for right in components[left_index + 1:]:
                for first in left:
                    for second in right:
                        if total >= cfg.total_cap:
                            return
                        if sum(name.startswith("contact:") for name in names) >= cfg.group_contact_cap:
                            return
                        if pinned[first] and pinned[second]:
                            continue
                        moved = first if pinned[second] else second
                        if not pinned[first] and not pinned[second]:
                            moved = max(first, second)
                        for axis in (0, 1):
                            for order in (0, 1):
                                candidate = _contact_candidate(contact_base, first, second,
                                                                axis, order, moved)
                                margin = min(float(candidate[first, 3 - axis]),
                                             float(candidate[second, 3 - axis]))
                                if not has_exact_positive_contact(candidate, first, second,
                                                                   axis, bool(order), margin):
                                    continue
                                if not matches_preplaced_origins(candidate, case):
                                    continue
                                name = f"contact:{gid}:{first}:{second}:{axis}:{order}"
                                emitted = _emit_candidate(name, candidate, "contact",
                                                           cfg.group_contact_cap, seen, names,
                                                           total, cfg.total_cap, cons)
                                if emitted is not None:
                                    yield emitted
                                    total += 1
                                    if total >= cfg.total_cap:
                                        return


def _validate_rects(rects: torch.Tensor) -> None:
    if not isinstance(rects, torch.Tensor) or not rects.is_floating_point() or rects.ndim != 3 or rects.shape[-1] != 4:
        raise ValueError("rects")
    if not torch.isfinite(rects).all() or (rects[..., 2:] <= 0).any():
        raise ValueError("rects values")


def _validate_batch(batch: SparseTopologyBatch, rects: torch.Tensor) -> None:
    if type(batch) is not SparseTopologyBatch:
        raise ValueError("batch")
    names = list(batch.__dataclass_fields__)
    if any(not isinstance(getattr(batch, n), torch.Tensor) or getattr(batch, n).ndim != 1 for n in names):
        raise ValueError("batch fields")
    if any(getattr(batch, n).device != rects.device for n in names):
        raise ValueError("device")
    for group in (names[:6], names[6:]):
        if len({getattr(batch, n).numel() for n in group}) != 1:
            raise ValueError("batch lengths")
    integer = {n for n in names if n.endswith(("batch", "src", "dst", "axis", "a", "b", "order"))}
    for n in names:
        value = getattr(batch, n)
        if n in integer:
            if value.dtype != torch.long:
                raise ValueError("integer field")
        elif not value.is_floating_point() or value.dtype != rects.dtype or not torch.isfinite(value).all():
            raise ValueError("float field")
        if n == "edge_margin" and (value < 0).any():
            raise ValueError("margin")
        if n == "contact_margin" and (value <= 0).any():
            raise ValueError("contact margin")
        if n.endswith("weight") and (value <= 0).any():
            raise ValueError("weight")
    b, n, _ = rects.shape
    for field, limit in (("edge_batch", b), ("contact_batch", b), ("edge_src", n), ("edge_dst", n), ("contact_a", n), ("contact_b", n)):
        value = getattr(batch, field)
        if (value < 0).any() or (value >= limit).any():
            raise ValueError("index")
    for field in ("edge_axis", "contact_axis", "contact_order"):
        value = getattr(batch, field)
        if (value < 0).any() or (value > 1).any():
            raise ValueError("range")
    if (batch.edge_src == batch.edge_dst).any() or (batch.contact_a == batch.contact_b).any():
        raise ValueError("self edge")


def topology_losses(rects: torch.Tensor, labels: SparseTopologyBatch, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    _validate_rects(rects)
    if not isinstance(scale, torch.Tensor) or not scale.is_floating_point() or scale.dtype != rects.dtype or scale.ndim != 1 or scale.shape[0] != rects.shape[0] or scale.device != rects.device or not torch.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("scale")
    _validate_batch(labels, rects)
    x, y, w, h = rects.unbind(-1)
    def mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if not values.numel():
            return rects.sum() * 0
        detached = weights.detach()
        return (values * detached).sum() / detached.sum().clamp_min(1e-12)
    eb, es, ed, ea = labels.edge_batch, labels.edge_src, labels.edge_dst, labels.edge_axis
    src = torch.stack((x[eb, es], y[eb, es]), -1); dst = torch.stack((x[eb, ed], y[eb, ed]), -1)
    size = torch.stack((w[eb, es], h[eb, es]), -1)
    gap = dst.gather(1, ea[:, None]).squeeze(1) - src.gather(1, ea[:, None]).squeeze(1) - size.gather(1, ea[:, None]).squeeze(1)
    separation = mean(torch.relu((labels.edge_margin - gap) / scale[eb].clamp_min(1e-6)), labels.edge_weight)
    cb, ca, cc, cax = labels.contact_batch, labels.contact_a, labels.contact_b, labels.contact_axis
    aa = torch.stack((x[cb, ca], y[cb, ca]), -1); bb = torch.stack((x[cb, cc], y[cb, cc]), -1)
    aw = torch.stack((w[cb, ca], h[cb, ca]), -1); bw = torch.stack((w[cb, cc], h[cb, cc]), -1)
    av = aa.gather(1, cax[:, None]).squeeze(1); bv = bb.gather(1, cax[:, None]).squeeze(1)
    asz = aw.gather(1, cax[:, None]).squeeze(1); bsz = bw.gather(1, cax[:, None]).squeeze(1)
    ordered_gap = torch.where(labels.contact_order.bool(), bv - av - asz, av - bv - bsz)
    perp = 1 - cax
    overlap = torch.minimum(aa.gather(1, perp[:, None]).squeeze(1) + aw.gather(1, perp[:, None]).squeeze(1), bb.gather(1, perp[:, None]).squeeze(1) + bw.gather(1, perp[:, None]).squeeze(1)) - torch.maximum(aa.gather(1, perp[:, None]).squeeze(1), bb.gather(1, perp[:, None]).squeeze(1))
    contact = (ordered_gap.abs() + torch.relu(labels.contact_margin - overlap)) / scale[cb].clamp_min(1e-6)
    contact_mean = mean(contact, labels.contact_weight)
    return {"separation": separation, "contact": contact_mean, "total": separation + contact_mean}


def _constraints(case: Mapping[str, Any], n: int) -> list[list[int]]:
    cons = case.get("cons")
    if not isinstance(cons, (list, tuple)) or len(cons) != n:
        raise ValueError("cons")
    widths = set()
    out = []
    for row in cons:
        if not isinstance(row, (list, tuple)) or len(row) not in (2, 5):
            raise ValueError("cons row")
        widths.add(len(row))
    if len(widths) > 1:
        raise ValueError("cons row")
    for row in cons:
        vals = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or int(value) != float(value):
                raise ValueError("cons value")
            vals.append(int(value))
        if vals[0] not in (0, 1) or vals[1] not in (0, 1) or (len(vals) == 5 and (vals[2] < 0 or vals[3] < 0 or not 0 <= vals[4] <= 15)):
            raise ValueError("cons value")
        out.append(vals)
    return out


def _separation_edges(r: list[list[float]], n: int) -> list[SparseEdge]:
    edges = []
    for axis in (0, 1):
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                gx = max(r[i][0] - r[j][0] - r[j][2], r[j][0] - r[i][0] - r[i][2])
                gy = max(r[i][1] - r[j][1] - r[j][3], r[j][1] - r[i][1] - r[i][3])
                if (axis == 0 and gx < gy) or (axis == 1 and gx >= gy):
                    continue
                ci = r[i][axis] + r[i][axis + 2] / 2; cj = r[j][axis] + r[j][axis + 2] / 2
                src, dst = (i, j) if (ci, i) <= (cj, j) else (j, i)
                candidates.append((src, dst, max(0.0, r[dst][axis] - r[src][axis] - r[src][axis + 2])))
        for src, dst, margin in candidates:
            reachable = set()
            frontier = [v for u, v, _ in candidates if u == src and v != dst]
            while frontier:
                node = frontier.pop()
                if node in reachable: continue
                reachable.add(node); frontier.extend(v for u, v, _ in candidates if u == node)
            if dst not in reachable:
                edges.append(SparseEdge(src, dst, axis, margin, "sep", 1.0))
    return edges


def extract_sparse_label(legal: torch.Tensor, case: Mapping[str, Any], instance_id: str, sample_seed: int, teacher_cost: float, base_cost: float) -> TopologyLabel:
    if not isinstance(case, Mapping): raise ValueError("case")
    if not isinstance(legal, torch.Tensor) or legal.device.type != "cpu" or legal.dtype != torch.float64 or legal.ndim != 2 or legal.shape[1] != 4: raise ValueError("legal")
    n = case.get("n")
    if type(n) is not int or n != legal.shape[0] or not isinstance(instance_id, str) or not instance_id.strip() or type(sample_seed) is not int or isinstance(sample_seed, bool): raise ValueError("metadata")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0 for v in (teacher_cost, base_cost)) or teacher_cost > base_cost: raise ValueError("cost")
    if not torch.isfinite(legal).all() or (legal[:, 2:] <= 0).any(): raise ValueError("legal values")
    r = legal.tolist(); cons = _constraints(case, n); edges = _separation_edges(r, n); contacts = []
    clusters: dict[int, list[int]] = {}
    for i, row in enumerate(cons):
        if len(row) == 5 and row[3] > 0: clusters.setdefault(row[3], []).append(i)
    for members in clusters.values():
        cand = []
        for ii, a in enumerate(members):
            for b in members[ii + 1:]:
                for axis in (0, 1):
                    perp = 1 - axis; a_end = r[a][axis] + r[a][axis + 2]; b_end = r[b][axis] + r[b][axis + 2]
                    if not (a_end == r[b][axis] or b_end == r[a][axis]): continue
                    overlap = min(r[a][perp] + r[a][perp + 2], r[b][perp] + r[b][perp + 2]) - max(r[a][perp], r[b][perp])
                    if overlap <= 0: continue
                    before = a_end == r[b][axis]
                    x, y = (a, b) if a < b else (b, a)
                    cand.append((-overlap, axis, x, y, before if x == a else not before, overlap))
        parent = {i: i for i in members}
        def find(i):
            while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for _, axis, a, b, order, overlap in sorted(cand):
            if find(a) != find(b): parent[find(a)] = find(b); contacts.append(ContactLabel(a, b, axis, bool(order), overlap, 1.0))
        if len({find(i) for i in members}) != 1: raise ValueError("disconnected cluster")
    incoming = {(e.axis, e.dst): [] for e in edges}
    for e in edges: incoming.setdefault((e.axis, e.dst), []).append(e.src)
    paths = []; pin_edges = []
    for axis in (0, 1):
        for target, row in enumerate(cons):
            if len(row) < 2 or row[1] != 1 or not incoming.get((axis, target)): continue
            chain = [target]; cur = target
            while incoming.get((axis, cur)):
                pred = min(incoming[(axis, cur)], key=lambda p: (-(r[p][axis] + r[p][axis + 2]), p))
                chain.append(pred); cur = pred
            chain.reverse(); paths.append(tuple(chain))
            for src, dst in zip(chain, chain[1:]):
                margin = next(e.margin for e in edges if e.axis == axis and e.src == src and e.dst == dst)
                pin_edges.append(SparseEdge(src, dst, axis, margin, "pin", 1.0))
    all_edges = tuple(sorted(set(edges + pin_edges), key=lambda e: (e.axis, e.src, e.dst, e.kind, e.margin, e.weight)))
    return TopologyLabel(instance_id, n, sample_seed, float(teacher_cost), float(base_cost), float(base_cost / teacher_cost), all_edges, tuple(sorted(set(contacts), key=lambda c: (c.axis, c.a, c.b))), tuple(sorted(set(paths))))
