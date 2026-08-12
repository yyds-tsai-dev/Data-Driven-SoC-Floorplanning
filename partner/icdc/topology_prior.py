"""Leak-free sparse topology labels and differentiable topology losses."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from . import tfdl as T
from . import engine


@dataclass(frozen=True)
class ProposalConfig:
    axis_exchange_cap: int = 8
    pin_repair_cap: int = 8
    group_contact_cap: int = 8
    total_cap: int = 32

    def __post_init__(self):
        for value in (self.axis_exchange_cap, self.pin_repair_cap, self.group_contact_cap, self.total_cap):
            if type(value) is not int or value < 0:
                raise TypeError("proposal cap")


@dataclass(frozen=True)
class ProposalResult:
    name: str
    rects: torch.Tensor
    legal: torch.Tensor
    drift: torch.Tensor
    hard_checks: Mapping[str, Any]
    cost: Any = None
    label: Any = None


def is_acyclic(n, edges):
    if type(n) is not int or n <= 0 or not isinstance(edges, (list, tuple)):
        return False
    graph = [[] for _ in range(n)]
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2 or any(type(x) is not int for x in edge): return False
        a, b = edge
        if not (0 <= a < n and 0 <= b < n) or a == b: return False
        graph[a].append(b)
    state = [0] * n
    def visit(node):
        if state[node] == 1: return False
        if state[node] == 2: return True
        state[node] = 1
        if any(not visit(child) for child in graph[node]): return False
        state[node] = 2
        return True
    return all(visit(i) for i in range(n))


def _rect_cpu(rects, n):
    return isinstance(rects, torch.Tensor) and rects.device.type == "cpu" and rects.is_floating_point() and rects.ndim in (2, 3) and rects.shape[-1] == 4 and (rects.shape[0] == n if rects.ndim == 2 else rects.shape[0] == 1 and rects.shape[1] == n) and torch.isfinite(rects).all()


def matches_preplaced_origins(rects, case):
    try:
        n = case.get("n"); cons = _constraints(case, n); tp = torch.as_tensor(case.get("tp"), dtype=torch.float64)
        if type(n) is not int or tp.shape != (n, 4) or not _rect_cpu(rects, n): return False
        value = rects[0] if rects.ndim == 3 else rects
        for i, row in enumerate(cons):
            if len(row) == 5 and row[1] and torch.isfinite(tp[i, :2]).all() and (tp[i, :2] >= 0).all() and not torch.equal(value[i, :2], tp[i, :2]): return False
        return True
    except Exception: return False


def has_exact_positive_contact(rects, a, b, axis, order, perp_margin):
    try:
        if not _rect_cpu(rects, rects.shape[-2]) or type(a) is not int or type(b) is not int or type(axis) is not int or type(order) is not bool or not isinstance(perp_margin, (int, float)) or isinstance(perp_margin, bool) or not math.isfinite(float(perp_margin)) or perp_margin <= 0 or a == b or axis not in (0, 1): return False
        r = rects[0] if rects.ndim == 3 else rects; n = r.shape[0]
        if not (0 <= a < n and 0 <= b < n): return False
        perp = 1 - axis; end_a = r[a, axis] + r[a, axis + 2]; end_b = r[b, axis] + r[b, axis + 2]
        if (end_a == r[b, axis]) != order: return False
        overlap = min(r[a, perp] + r[a, perp + 2], r[b, perp] + r[b, perp + 2]) - max(r[a, perp], r[b, perp])
        return bool(overlap >= perp_margin and overlap > 0)
    except Exception: return False


def pin_feasible_then_exact_tfdl(seed, case):
    try:
        n = case.get("n"); cons = _constraints(case, n); area = torch.as_tensor(case.get("area"), dtype=torch.float64)
        tp = torch.as_tensor(case.get("tp"), dtype=torch.float64)
        if type(n) is not int or area.shape != (n,) or tp.shape != (n, 4) or not _rect_cpu(seed, n): return None
        if not torch.isfinite(area).all() or (area <= 0).any() or not torch.isfinite(tp[tp >= 0]).all(): return None
        original = seed.to(torch.float64).clone().unsqueeze(0)
        cons_t = torch.tensor(cons, dtype=torch.long).unsqueeze(0)
        mask = area.unsqueeze(0) > 0
        pinned = (cons_t[:, :, 1] != 0) & (tp[:, 0] >= 0).unsqueeze(0) & mask
        pin_xy = tp[:, :2].unsqueeze(0)
        boundary = torch.tensor([[r[4] if len(r) == 5 else 0 for r in cons]], dtype=torch.long)
        def run(exact):
            legal, drift = T.tfdl(original.clone(), mask, pinned, pin_xy=pin_xy, boundary_code=boundary, exact=exact)
            if not isinstance(legal, torch.Tensor) or not isinstance(drift, torch.Tensor) or legal.shape != original.shape or drift.shape != (1, n, 2) or not torch.isfinite(legal).all() or not torch.isfinite(drift).all() or torch.count_nonzero(drift) != 0 or not matches_preplaced_origins(legal, case): return None
            return legal, drift
        if run(False) is None: return None
        result = run(True)
        if result is None: return None
        legal, drift = result
        checks = engine.verify_hard_legal(legal.squeeze(0).numpy(), area.numpy(), __import__("numpy").asarray(cons), tp.numpy())
        if not isinstance(checks, Mapping) or not checks or not all(type(v) is bool or isinstance(v, __import__("numpy").bool_) for v in checks.values()) or not all(checks.values()): return None
        return legal.squeeze(0), drift.squeeze(0)
    except Exception: return None


def _proposal_fingerprint(rects, cons):
    r = rects.tolist(); out = []
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            gx = max(r[i][0] - r[j][0] - r[j][2], r[j][0] - r[i][0] - r[i][2])
            gy = max(r[i][1] - r[j][1] - r[j][3], r[j][1] - r[i][1] - r[i][3])
            axis = 0 if gx >= gy else 1; ci = r[i][axis] + r[i][axis + 2] / 2; cj = r[j][axis] + r[j][axis + 2] / 2
            out.append((i, j, axis, (ci, i) <= (cj, j)))
            if len(cons[i]) == 5 and cons[i][3] == cons[j][3] and cons[i][3] > 0:
                perp = 1 - axis; overlap = min(r[i][perp] + r[i][perp + 2], r[j][perp] + r[j][perp + 2]) - max(r[i][perp], r[j][perp])
                if overlap > 0 and (r[i][axis] + r[i][axis + 2] == r[j][axis] or r[j][axis] + r[j][axis + 2] == r[i][axis]): out.append(("c", i, j, axis, overlap))
    return tuple(out)


def generate_proposals(raw, case, cfg=ProposalConfig()):
    if type(cfg) is not ProposalConfig or not isinstance(raw, torch.Tensor) or raw.device.type != "cpu" or raw.ndim != 2 or raw.shape[1] != 4 or not raw.is_floating_point() or not torch.isfinite(raw).all() or (raw[:, 2:] <= 0).any(): raise ValueError("raw")
    n = case.get("n"); cons = _constraints(case, n)
    if type(n) is not int or raw.shape[0] != n: raise ValueError("case")
    base = raw.to(torch.float64).clone(); seen = {_proposal_fingerprint(base, cons)}; emitted = 0
    if cfg.total_cap <= 0: return
    yield "base", base
    emitted += 1
    for kind, cap in (("axis", cfg.axis_exchange_cap), ("pin", cfg.pin_repair_cap), ("contact", cfg.group_contact_cap)):
        for i in range(n):
            if emitted >= cfg.total_cap or cap <= 0: break
            candidate = base.clone(); target = (i + 1) % n; axis = 0
            if kind == "axis" and len(cons[target]) >= 2 and cons[target][1] == 1:
                continue
            if kind == "contact" and len(cons[i]) == 5 and cons[i][3] == cons[target][3] and cons[i][3] > 0:
                candidate[target, 0] = candidate[i, 0] + candidate[i, 2]
            elif kind == "pin":
                if cons[i][1] == 1: continue
                candidate[target, 0] = candidate[i, 0] + candidate[i, 2] + 1e-6
            else: candidate[target, axis] = candidate[i, 0] + candidate[i, 2] + 1e-6
            fp = _proposal_fingerprint(candidate, cons)
            if fp in seen: continue
            seen.add(fp); yield f"{kind}:{i}:{target}", candidate; emitted += 1

from .topology_data import ContactLabel, SparseEdge, SparseTopologyBatch, TopologyLabel


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
