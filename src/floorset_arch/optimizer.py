from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from floorset_arch.constructive import construct_beam_placement
from floorset_arch.diagnostics import placement_metrics, repair_delta
from floorset_arch.geometry import bbox, boundary_satisfied, edge_touch_length
from floorset_arch.hetero_graph import build_hetero_floorplan_graph
from floorset_arch.models import AnchorGuidance, Placement, Rect, SolverConfig
from floorset_arch.parser import parse_instance
from floorset_arch.relative_order import construct_relative_order_placement
from floorset_arch.repair import repair_placement
from floorset_arch.scoring import hpwl_proxy


ROOT = Path(__file__).resolve().parents[2]
CONTEST_DIR = ROOT / "FloorSet" / "iccad2026contest"
if CONTEST_DIR.exists() and str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))

try:
    from iccad2026_evaluate import FloorplanOptimizer
except Exception:
    class FloorplanOptimizer:  # type: ignore[no-redef]
        def __init__(self, verbose: bool = False):
            self.verbose = verbose


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    profile: str
    kind: str = "relative_order"
    repair_profile: str = "normal"
    disable_guidance: bool = False


def _repair_with_profile_worker(inst, placement: Placement, config: SolverConfig, repair_profile: str) -> Placement:
    if repair_profile != "large_boundary":
        return repair_placement(inst, placement, config)

    saved_pass = os.environ.get("FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_PASS")
    saved_cap = os.environ.get("FLOORSET_LARGE_CASE_BOUNDARY_AXIS_CAP")
    os.environ["FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_PASS"] = "1"
    os.environ.setdefault("FLOORSET_LARGE_CASE_BOUNDARY_AXIS_CAP", "160")
    try:
        return repair_placement(inst, placement, config)
    finally:
        if saved_pass is None:
            os.environ.pop("FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_PASS", None)
        else:
            os.environ["FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_PASS"] = saved_pass
        if saved_cap is None:
            os.environ.pop("FLOORSET_LARGE_CASE_BOUNDARY_AXIS_CAP", None)
        else:
            os.environ["FLOORSET_LARGE_CASE_BOUNDARY_AXIS_CAP"] = saved_cap


def _build_candidate_worker(inst, config: SolverConfig, spec: CandidateSpec) -> Placement:
    saved_guidance = inst.anchor_guidance
    if spec.disable_guidance:
        inst.anchor_guidance = None
    try:
        if spec.kind == "beam":
            graph = build_hetero_floorplan_graph(inst)
            placement = construct_beam_placement(inst, config, graph=graph)
        else:
            placement = construct_relative_order_placement(inst, config, profile=spec.profile)
        return _repair_with_profile_worker(inst, placement, config, spec.repair_profile)
    finally:
        if spec.disable_guidance:
            inst.anchor_guidance = saved_guidance


class ArchitectureV3Optimizer(FloorplanOptimizer):
    """Anchor-GNN guided hetero-graph beam solver."""

    def __init__(self, verbose: bool = False, config: Optional[SolverConfig] = None):
        super().__init__(verbose=verbose)
        self.config = config or SolverConfig()
        if os.environ.get("FLOORSET_BEAM_WIDTH"):
            try:
                self.config.beam_width = max(1, int(os.environ["FLOORSET_BEAM_WIDTH"]))
            except ValueError:
                pass
        self._checkpoint_key: Optional[Path] = None
        self._checkpoint_model = None
        self._checkpoint_config: dict = {}
        self._checkpoint_kind = ""

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:
        inst = parse_instance(
            block_count,
            area_targets,
            b2b_connectivity,
            p2b_connectivity,
            pins_pos,
            constraints,
            target_positions,
        )
        inst.anchor_guidance = self._try_anchor_guidance(inst)
        candidates = self._build_candidates(inst, self._candidate_specs(inst))
        best = self._select_best_candidate(inst, candidates)
        return best.to_position_list(block_count)

    def _build_candidates(self, inst, specs: list[CandidateSpec]) -> list[Placement]:
        workers = self._candidate_workers(len(specs))
        if workers <= 1:
            return [self._build_candidate(inst, spec) for spec in specs]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._build_candidate, [inst] * len(specs), specs))

    def _candidate_workers(self, candidate_count: int) -> int:
        if candidate_count <= 1:
            return 1
        raw = os.environ.get("FLOORSET_CANDIDATE_WORKERS", "1")
        try:
            requested = int(raw)
        except ValueError:
            return 1
        return max(1, min(requested, candidate_count))

    def _candidate_specs(self, inst) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        profile_policy = os.environ.get("FLOORSET_PROFILE_POLICY", "adaptive").strip().lower()
        include_compact = os.environ.get("FLOORSET_INCLUDE_COMPACT_RELATIVE", "1") == "1"

        if profile_policy == "compact":
            profile_order = ["compact"] if include_compact else []
        elif profile_policy == "soft":
            profile_order = ["soft"]
        elif profile_policy == "both":
            profile_order = ["soft", "compact"] if include_compact else ["soft"]
        elif inst.block_count >= 118 and os.environ.get("FLOORSET_ENABLE_LARGE_CASE_CANDIDATES", "0") == "1":
            adaptive_profile = self._adaptive_profile_order(inst)[0]
            repair_profiles = self._large_case_repair_profiles()
            for repair_profile in repair_profiles:
                seen_profiles: set[str] = set()
                for name, profile in (
                    ("adaptive_relative_order", adaptive_profile),
                    ("forced_soft_relative_order", "soft"),
                    ("forced_compact_relative_order", "compact"),
                ):
                    if profile == "compact" and not include_compact:
                        continue
                    if profile in seen_profiles:
                        continue
                    seen_profiles.add(profile)
                    specs.append(
                        CandidateSpec(
                            name=name,
                            profile=profile,
                            repair_profile=repair_profile,
                        )
                    )
            return specs
        else:
            profile_order = self._adaptive_profile_order(inst)

        for profile in profile_order:
            if profile == "compact" and not include_compact:
                continue
            specs.append(CandidateSpec(name=f"{profile}_relative_order", profile=profile))
        if not specs:
            specs.append(CandidateSpec(name="fallback_soft_relative_order", profile="soft"))

        include_beam = os.environ.get("FLOORSET_INCLUDE_BEAM_CANDIDATES", "0") == "1"
        if include_beam:
            specs.append(CandidateSpec(name="beam", profile="beam", kind="beam"))

        include_no_guidance = os.environ.get("FLOORSET_INCLUDE_NO_GUIDANCE_CANDIDATE", "0") == "1"
        if include_no_guidance and inst.anchor_guidance is not None:
            specs.append(
                CandidateSpec(
                    name="no_guidance_relative_order",
                    profile="soft",
                    kind="relative_order",
                    disable_guidance=True,
                )
            )
            if include_beam:
                specs.append(
                    CandidateSpec(
                        name="no_guidance_beam",
                        profile="beam",
                        kind="beam",
                        disable_guidance=True,
                    )
                )
        return specs

    def _large_case_repair_profiles(self) -> list[str]:
        raw = os.environ.get("FLOORSET_LARGE_CASE_REPAIR_PROFILES", "normal")
        profiles = [part.strip() for part in raw.split(",") if part.strip()]
        allowed = {"normal", "large_boundary"}
        filtered = [profile for profile in profiles if profile in allowed]
        return filtered or ["normal"]

    def _build_candidate(self, inst, spec: CandidateSpec) -> Placement:
        saved_guidance = inst.anchor_guidance
        if spec.disable_guidance:
            inst.anchor_guidance = None
        try:
            if spec.kind == "beam":
                graph = build_hetero_floorplan_graph(inst)
                placement = construct_beam_placement(inst, self.config, graph=graph)
            else:
                placement = construct_relative_order_placement(inst, self.config, profile=spec.profile)
            return self._repair_candidate(inst, placement, spec)
        finally:
            if spec.disable_guidance:
                inst.anchor_guidance = saved_guidance

    def _repair_candidate(self, inst, placement: Placement, spec: CandidateSpec) -> Placement:
        before = placement
        after = self._repair_with_profile(inst, before, spec.repair_profile)
        self._trace_repair(
            inst,
            before,
            after,
            {
                "name": spec.name,
                "profile": spec.profile,
                "kind": spec.kind,
                "repair_profile": spec.repair_profile,
            },
        )
        return after

    def _repair_with_profile(self, inst, placement: Placement, repair_profile: str) -> Placement:
        return _repair_with_profile_worker(inst, placement, self.config, repair_profile)

    def _select_best_candidate(self, inst, candidates: list[Placement]) -> Placement:
        return min(candidates, key=lambda placement: self._candidate_rank(inst, placement))

    def _candidate_rank(self, inst, placement: Placement) -> tuple[int, int, int, float]:
        metrics = placement_metrics(inst, placement)
        boundary = int(metrics["boundary_violations"])
        grouping = int(metrics["group_violations"])
        mib = int(metrics["mib_violations"])
        return boundary + grouping + mib, boundary, grouping, self._proxy_cost(inst, placement)

    def _trace_repair(self, inst, before: Placement, after: Placement, candidate: dict[str, str]) -> None:
        trace_path = os.environ.get("FLOORSET_REPAIR_TRACE_JSONL")
        if not trace_path:
            return
        before_metrics = placement_metrics(inst, before)
        after_metrics = placement_metrics(inst, after)
        row = {
            "block_count": inst.block_count,
            "checkpoint_loaded": self._checkpoint_model is not None,
            "candidate": candidate,
            "before_repair": before_metrics,
            "after_repair": after_metrics,
            "delta": repair_delta(before, after, before_metrics, after_metrics),
        }
        with open(trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _try_anchor_guidance(self, inst) -> Optional[AnchorGuidance]:
        checkpoint = self._resolve_checkpoint_path(os.environ.get(self.config.checkpoint_env))
        if not checkpoint:
            return None
        try:
            from floorset_arch.features import build_anchor_edge_tensors, build_anchor_node_features
            from floorset_arch.nn.model import FloorplanGNN

            if self._checkpoint_key != checkpoint or self._checkpoint_model is None:
                from floorset_arch.training import checkpoint as checkpoint_io

                payload = checkpoint_io.load_checkpoint(checkpoint, map_location="cpu")
                if "model_state_dict" not in payload:
                    return None
                self._checkpoint_kind = "anchor_gnn_v2"
                model = FloorplanGNN(
                    node_feat_dim=int(payload["node_feat_dim"]),
                    hidden_dim=int(payload.get("hidden_dim", 160)),
                    num_layers=int(payload.get("layers", 5)),
                    dropout=float(payload.get("dropout", 0.05)),
                )
                model.load_state_dict(payload["model_state_dict"], strict=False)
                self._checkpoint_config = {
                    "node_feat_dim": int(payload["node_feat_dim"]),
                    "hidden_dim": int(payload.get("hidden_dim", 160)),
                    "layers": int(payload.get("layers", 5)),
                    "dropout": float(payload.get("dropout", 0.05)),
                    "has_pair_head": bool(payload.get("has_pair_head", False)),
                }
                model.eval()
                self._checkpoint_model = model
                self._checkpoint_key = checkpoint
            else:
                model = self._checkpoint_model

            with torch.no_grad():
                node_feat, scale = build_anchor_node_features(inst, device=torch.device("cpu"))
                if node_feat.shape[1] != int(self._checkpoint_config.get("node_feat_dim", node_feat.shape[1])):
                    return None
                edge_index, edge_attr = build_anchor_edge_tensors(inst, device=torch.device("cpu"))
                pairs = None
                if self._checkpoint_config.get("has_pair_head"):
                    pairs = torch.tensor(
                        [(i, j) for i in range(inst.block_count) for j in range(i + 1, inst.block_count)],
                        dtype=torch.long,
                    )
                pred = model(node_feat, edge_index, edge_attr, pairs=pairs)
                return self._anchor_predictions_to_guidance(inst, pred, scale, pairs)
        except Exception:
            return None

    def _anchor_predictions_to_guidance(
        self,
        inst,
        pred: dict[str, torch.Tensor],
        scale: float,
        pairs: torch.Tensor | None = None,
    ) -> AnchorGuidance:
        anchors = pred["anchor"].detach().cpu() * max(float(scale), 1.0)
        priority_tensor = pred.get("priority")
        aspect_tensor = pred.get("log_aspect")
        guidance = AnchorGuidance(scale=max(float(scale), 1.0))
        for i in range(inst.block_count):
            target = inst.target_rects.get(i)
            if target is not None and i in inst.preplaced:
                guidance.rect_priors[i] = target
                continue
            if target is not None and i in inst.fixed:
                width, height = target.width, target.height
            else:
                area = max(1.0, float(inst.area_targets[i]))
                width = math.sqrt(area)
                height = math.sqrt(area)
            if aspect_tensor is not None and i not in inst.fixed and i not in inst.preplaced:
                log_aspect = float(aspect_tensor.detach().cpu()[i])
                guidance.log_aspect[i] = log_aspect
                aspect = math.exp(max(-2.5, min(2.5, log_aspect)))
                area = max(1.0, float(inst.area_targets[i]))
                width = math.sqrt(area * aspect)
                height = math.sqrt(area / aspect)
            cx = float(anchors[i, 0])
            cy = float(anchors[i, 1])
            if math.isfinite(cx) and math.isfinite(cy):
                guidance.rect_priors[i] = Rect(max(0.0, cx - width / 2.0), max(0.0, cy - height / 2.0), width, height)
            if priority_tensor is not None:
                guidance.priority[i] = float(priority_tensor.detach().cpu()[i])
        pair_logits = pred.get("pair_logits")
        if pair_logits is not None and pairs is not None:
            logits = pair_logits.detach().cpu()
            for pair, logit in zip(pairs.detach().cpu().tolist(), logits.tolist()):
                guidance.pairwise_axis[(int(pair[0]), int(pair[1]))] = (float(logit[0]), float(logit[1]))
        return guidance

    def _resolve_checkpoint_path(self, value: Optional[str]) -> Optional[Path]:
        if not value:
            value = self.config.default_checkpoint
        raw = Path(value).expanduser()
        candidates = [raw]
        if not raw.is_absolute() and self.config.checkpoint_repo_relative:
            candidates.extend([ROOT / raw, ROOT / "checkpoints" / raw.name])
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()

    def _adaptive_profile_order(self, inst) -> list[str]:
        """Pick one relative-order profile for runtime-heavy cases.

        Computing both soft and compact profiles is useful for ablations, but the
        official score applies a runtime factor and the 100+ block cases dominate
        the weighted total.  The features below are cheap instance statistics that
        separate pin-heavy compact wins from net-heavy / boundary-sensitive soft
        wins without touching the evaluator loop.
        """

        if inst.block_count < 90:
            return ["soft", "compact"]

        b2b_count = int(inst.valid_b2b.shape[0]) if inst.valid_b2b is not None else 0
        p2b_count = int(inst.valid_p2b.shape[0]) if inst.valid_p2b is not None else 0
        cluster_count = len(inst.cluster_groups)
        max_mib = max((len(members) for members in inst.mib_groups.values()), default=0)
        fixed_count = len(inst.fixed)
        preplaced_count = len(inst.preplaced)
        boundary_count = len(inst.boundary)

        if p2b_count > 2500:
            return ["soft"] if b2b_count > 5000 else ["compact"]
        if b2b_count > 5000:
            return ["soft"] if p2b_count < 200 else ["compact"]
        if cluster_count <= 3:
            if p2b_count < 700 and b2b_count < 1300 and preplaced_count <= 3:
                return ["soft"]
            return ["compact"]
        if p2b_count < 100:
            return ["soft"]
        if preplaced_count >= 5 and boundary_count <= 30:
            return ["soft"]
        if max_mib >= 7 and preplaced_count < 5:
            return ["compact"]
        if fixed_count >= 14:
            return ["compact"] if p2b_count > 800 else ["soft"]
        if boundary_count <= 28:
            return ["soft"]
        return ["compact"]

    def _proxy_cost(self, inst, placement: Placement) -> float:
        rects = placement.rects
        bounds = bbox(list(rects.values()))
        hpwl = hpwl_proxy(inst, rects)
        total_area = float(torch.clamp(inst.area_targets[: inst.block_count], min=1.0).sum().item())
        edge_weight = sum(float(w) for *_ij, w in inst.valid_b2b.tolist()) + sum(
            float(w) for *_ij, w in inst.valid_p2b.tolist()
        )
        hpwl_scale = max(1.0, edge_weight * max(1.0, total_area**0.5))
        area_score = bounds.area / max(total_area, 1.0)
        hpwl_score = hpwl / hpwl_scale
        boundary_violations = 0
        for block, code in inst.boundary.items():
            rect = rects.get(block)
            if rect is None or not boundary_satisfied(rect, bounds, code):
                boundary_violations += 1
        group_violations = 0
        for members in inst.cluster_groups.values():
            present = [block for block in members if block in rects]
            if len(present) <= 1:
                continue
            seen = {present[0]}
            stack = [present[0]]
            while stack:
                cur = stack.pop()
                for other in present:
                    if other not in seen and edge_touch_length(rects[cur], rects[other]) > 0:
                        seen.add(other)
                        stack.append(other)
            group_violations += len(present) - len(seen)
        mib_violations = 0
        for members in inst.mib_groups.values():
            shapes = {
                (round(rects[block].width, 4), round(rects[block].height, 4))
                for block in members
                if block in rects
            }
            mib_violations += max(0, len(shapes) - 1)
        n_soft = max(1, len(inst.boundary))
        n_soft += sum(max(0, len(members) - 1) for members in inst.cluster_groups.values())
        n_soft += sum(max(0, len(members) - 1) for members in inst.mib_groups.values())
        v_rel = (boundary_violations + group_violations + mib_violations) / n_soft
        return (1.0 + 0.5 * (hpwl_score + area_score)) * (2.718281828 ** (2.0 * v_rel))
