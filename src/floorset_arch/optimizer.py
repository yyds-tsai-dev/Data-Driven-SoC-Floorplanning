from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from floorset_arch.constructive import construct_beam_placement
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


class ArchitectureV2Optimizer(FloorplanOptimizer):
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
        candidates: list[Placement] = []
        candidates.append(repair_placement(inst, construct_relative_order_placement(inst, self.config), self.config))
        include_compact = os.environ.get("FLOORSET_INCLUDE_COMPACT_RELATIVE", "1") == "1"
        if include_compact:
            candidates.append(
                repair_placement(
                    inst,
                    construct_relative_order_placement(inst, self.config, profile="compact"),
                    self.config,
                )
            )
        include_beam = os.environ.get("FLOORSET_INCLUDE_BEAM_CANDIDATES", "0") == "1"
        if include_beam:
            graph = build_hetero_floorplan_graph(inst)
            candidates.append(repair_placement(inst, construct_beam_placement(inst, self.config, graph=graph), self.config))

        include_no_guidance = os.environ.get("FLOORSET_INCLUDE_NO_GUIDANCE_CANDIDATE", "0") == "1"
        if include_no_guidance and inst.anchor_guidance is not None:
            saved_guidance = inst.anchor_guidance
            inst.anchor_guidance = None
            candidates.append(repair_placement(inst, construct_relative_order_placement(inst, self.config), self.config))
            if include_beam:
                graph_no_guidance = build_hetero_floorplan_graph(inst)
                candidates.append(repair_placement(inst, construct_beam_placement(inst, self.config, graph=graph_no_guidance), self.config))
            inst.anchor_guidance = saved_guidance

        best = min(candidates, key=lambda placement: self._proxy_cost(inst, placement))
        return best.to_position_list(block_count)

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
                )
                model.load_state_dict(payload["model_state_dict"])
                self._checkpoint_config = {
                    "node_feat_dim": int(payload["node_feat_dim"]),
                    "hidden_dim": int(payload.get("hidden_dim", 160)),
                    "layers": int(payload.get("layers", 5)),
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
                pred = model(node_feat, edge_index, edge_attr)
                return self._anchor_predictions_to_guidance(inst, pred, scale)
        except Exception:
            return None

    def _anchor_predictions_to_guidance(
        self,
        inst,
        pred: dict[str, torch.Tensor],
        scale: float,
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
