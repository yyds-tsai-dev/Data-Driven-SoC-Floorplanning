from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import torch

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is present in local runs.
    load_dotenv = None

from floorset_arch.constructive import construct_beam_placement
from floorset_arch.diagnostics import placement_metrics, repair_delta
from floorset_arch.geometry import (
    candidate_frontier_points,
    first_non_overlapping,
)
from floorset_arch.hetero_graph import build_hetero_floorplan_graph
from floorset_arch.models import AnchorGuidance, Placement, Rect, SolverConfig
from floorset_arch.parser import parse_instance
from floorset_arch.quality_portfolio import (
    enabled_quality_profiles,
    is_quality_portfolio_case,
    refine_quality_candidate,
)
from floorset_arch.relative_order import construct_relative_order_placement
from floorset_arch.repair import _runtime_tail_clamped_config, repair_placement
from floorset_arch.risk_budget import BudgetTier, instance_risk_budget
from floorset_arch.surrogate_guidance import build_surrogate_guidance
from floorset_arch.v10_proxy import v10_proxy_better, v10_proxy_cost, v10_proxy_rank

ROOT = Path(__file__).resolve().parents[2]
CONTEST_DIR = ROOT / "FloorSet" / "iccad2026contest"
if CONTEST_DIR.exists() and str(CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(CONTEST_DIR))


def _load_repo_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env", override=False)


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
    quality_profile: str = "default"


def _repair_with_profile_worker(
    inst, placement: Placement, config: SolverConfig, repair_profile: str
) -> Placement:
    profile_config = _repair_profile_config(config, repair_profile, inst)
    if repair_profile not in {"large_boundary", "boundary_first", "quality_refine"}:
        return repair_placement(inst, placement, profile_config)

    saved_pass = os.environ.get("FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_PASS")
    saved_refine = os.environ.get("FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_REFINE")
    saved_cap = os.environ.get("FLOORSET_LARGE_CASE_BOUNDARY_AXIS_CAP")
    saved_overlap = os.environ.get("FLOORSET_OVERLAP_REPAIR_CANDIDATES")
    if repair_profile in {"large_boundary", "boundary_first"}:
        os.environ["FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_PASS"] = "1"
        os.environ.setdefault("FLOORSET_LARGE_CASE_BOUNDARY_AXIS_CAP", "160")
    if repair_profile == "quality_refine":
        os.environ.setdefault("FLOORSET_OVERLAP_REPAIR_CANDIDATES", "64")
    try:
        return repair_placement(inst, placement, profile_config)
    finally:
        if saved_pass is None:
            os.environ.pop("FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_PASS", None)
        else:
            os.environ["FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_PASS"] = saved_pass
        if saved_refine is None:
            os.environ.pop("FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_REFINE", None)
        else:
            os.environ["FLOORSET_ENABLE_LARGE_CASE_BOUNDARY_REFINE"] = saved_refine
        if saved_cap is None:
            os.environ.pop("FLOORSET_LARGE_CASE_BOUNDARY_AXIS_CAP", None)
        else:
            os.environ["FLOORSET_LARGE_CASE_BOUNDARY_AXIS_CAP"] = saved_cap
        if saved_overlap is None:
            os.environ.pop("FLOORSET_OVERLAP_REPAIR_CANDIDATES", None)
        else:
            os.environ["FLOORSET_OVERLAP_REPAIR_CANDIDATES"] = saved_overlap


def _repair_profile_config(config: SolverConfig, repair_profile: str, inst=None) -> SolverConfig:
    if repair_profile in {"large_boundary", "boundary_first"}:
        profile_config = replace(
            config,
            max_boundary_component_snaps=max(config.max_boundary_component_snaps, 48),
            max_repair_passes=max(config.max_repair_passes, 10),
            soft_proxy_slack=max(config.soft_proxy_slack, 0.65),
        )
    elif repair_profile == "grouping_first":
        profile_config = replace(
            config,
            max_cluster_component_moves=max(config.max_cluster_component_moves, 56),
            max_pair_candidates_per_component=max(
                config.max_pair_candidates_per_component, 96
            ),
            soft_proxy_slack=max(config.soft_proxy_slack, 0.62),
        )
    elif repair_profile == "quality_refine":
        profile_config = replace(
            config,
            equal_soft_proxy_slack=max(config.equal_soft_proxy_slack, 0.04),
            max_repair_passes=max(config.max_repair_passes, 10),
        )
    else:
        profile_config = config
    if inst is None:
        return profile_config
    return _runtime_tail_clamped_config(inst, profile_config)


def _build_candidate_worker(
    inst, config: SolverConfig, spec: CandidateSpec
) -> Placement:
    saved_guidance = inst.anchor_guidance
    if spec.disable_guidance:
        inst.anchor_guidance = None
    try:
        if spec.kind == "beam":
            graph = build_hetero_floorplan_graph(inst)
            placement = construct_beam_placement(inst, config, graph=graph)
        else:
            placement = construct_relative_order_placement(
                inst, config, profile=spec.profile
            )
        return _repair_with_profile_worker(inst, placement, config, spec.repair_profile)
    finally:
        if spec.disable_guidance:
            inst.anchor_guidance = saved_guidance


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


class ArchitectureV5Optimizer(FloorplanOptimizer):
    """Anchor-GNN guided hetero-graph beam solver with selectable checkpoint encoders."""

    def __init__(self, verbose: bool = False, config: Optional[SolverConfig] = None):
        super().__init__(verbose=verbose)
        _load_repo_dotenv()
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
        if inst.anchor_guidance is None and self._uses_surrogate_guidance():
            inst.anchor_guidance = build_surrogate_guidance(inst)
        candidates = self._build_candidates(inst, self._candidate_specs(inst))
        best = self._select_best_candidate(inst, candidates)
        return best.to_position_list(block_count)

    def _uses_surrogate_guidance(self) -> bool:
        mode = os.environ.get("FLOORSET_ENABLE_SURROGATE_GUIDANCE", "0").strip().lower()
        return mode not in {"0", "false", "off", "no"}

    def _build_candidates(self, inst, specs: list[CandidateSpec]) -> list[Placement]:
        workers = self._candidate_worker_count_for_specs(specs)
        budget_stopped = False
        if workers <= 1:
            candidates = []
            for spec in specs:
                candidate = self._build_candidate(inst, spec)
                candidates.append(candidate)
                if self._conditional_runtime_budget_stops_expansion(candidate):
                    budget_stopped = True
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                candidates = list(
                    pool.map(self._build_candidate, [inst] * len(specs), specs)
                )
        if budget_stopped:
            return candidates
        return self._with_quality_refined_candidates(inst, candidates)

    def _conditional_runtime_budget_stops_expansion(self, placement: Placement) -> bool:
        from floorset_arch.budget_layer import (
            conditional_runtime_budget_enabled,
            conditional_runtime_budget_limits,
        )

        if not conditional_runtime_budget_enabled():
            return False
        trace = getattr(placement, "runtime_budget_trace", None)
        if not isinstance(trace, dict):
            return False
        try:
            elapsed_ms = float(trace.get("elapsed_ms", 0.0))
        except (TypeError, ValueError):
            return False
        stop_reason = str(trace.get("stop_reason", ""))
        if stop_reason == "elapsed_budget":
            return True
        try:
            tier = BudgetTier(str(trace.get("tier", BudgetTier.LIGHT.value)))
        except ValueError:
            tier = BudgetTier.LIGHT
        limits = conditional_runtime_budget_limits(tier)
        return (
            stop_reason == "rejected_attempts"
            or elapsed_ms >= limits.max_elapsed_ms
        )

    def _candidate_worker_count_for_specs(self, specs: list[CandidateSpec]) -> int:
        if any(spec.repair_profile != "normal" for spec in specs):
            return 1
        return self._candidate_workers(len(specs))

    def _quality_refine_worker_count(self, job_count: int) -> int:
        if job_count <= 1:
            return 1
        raw = (
            os.environ.get("FLOORSET_QUALITY_PORTFOLIO_WORKERS", "auto").strip().lower()
        )
        if raw == "auto":
            cpu_count = os.cpu_count() or 1
            return max(1, min(job_count, 2, cpu_count))
        try:
            requested = int(raw)
        except ValueError:
            return 1
        return max(1, min(requested, job_count, os.cpu_count() or 1))

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
        profile_policy = (
            os.environ.get("FLOORSET_PROFILE_POLICY", "adaptive").strip().lower()
        )
        include_compact = (
            os.environ.get("FLOORSET_INCLUDE_COMPACT_RELATIVE", "1") == "1"
        )
        high_risk_portfolio = self._uses_high_risk_portfolio(inst)

        if profile_policy == "compact":
            profile_order = ["compact"] if include_compact else []
        elif profile_policy == "soft":
            profile_order = ["soft"]
        elif profile_policy == "both":
            profile_order = ["soft", "compact"] if include_compact else ["soft"]
        elif high_risk_portfolio:
            adaptive_profile = self._adaptive_profile_order(inst)[0]
            profile_order = [adaptive_profile]
            default_shape_profiles = (
                "soft,compact,wide,tall"
                if self._uses_dense_shape_profiles(inst)
                else "soft,compact"
            )
            shape_profiles = os.environ.get(
                "FLOORSET_HIGH_RISK_SHAPE_PROFILES", default_shape_profiles
            )
            for profile in [
                part.strip() for part in shape_profiles.split(",") if part.strip()
            ]:
                if profile not in profile_order:
                    profile_order.append(profile)
            repair_profiles = self._high_risk_repair_profiles()
            for repair_profile in repair_profiles:
                for profile in profile_order:
                    if profile == "compact" and not include_compact:
                        continue
                    specs.append(
                        CandidateSpec(
                            name=f"high_risk_{profile}_{repair_profile}",
                            profile=profile,
                            repair_profile=repair_profile,
                        )
                    )
            return specs
        elif (
            inst.block_count >= 118
            and os.environ.get("FLOORSET_ENABLE_LARGE_CASE_CANDIDATES", "0") == "1"
        ):
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
            specs.append(
                CandidateSpec(name=f"{profile}_relative_order", profile=profile)
            )
        if not specs:
            specs.append(
                CandidateSpec(name="fallback_soft_relative_order", profile="soft")
            )

        include_beam = os.environ.get("FLOORSET_INCLUDE_BEAM_CANDIDATES", "0") == "1"
        if include_beam:
            specs.append(CandidateSpec(name="beam", profile="beam", kind="beam"))

        include_no_guidance = (
            os.environ.get("FLOORSET_INCLUDE_NO_GUIDANCE_CANDIDATE", "0") == "1"
        )
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

    def _with_quality_refined_candidates(
        self, inst, candidates: list[Placement]
    ) -> list[Placement]:
        if not candidates or not self._uses_quality_portfolio(inst):
            return candidates
        profiles = [
            profile for profile in self._quality_profiles() if profile != "default"
        ]
        if not profiles:
            return candidates
        base_limit = max(
            1, int(os.environ.get("FLOORSET_QUALITY_PORTFOLIO_BASE_LIMIT", "1"))
        )
        seeds = sorted(
            candidates, key=lambda placement: self._candidate_rank(inst, placement)
        )[:base_limit]
        jobs = [(seed, profile) for seed in seeds for profile in profiles]
        workers = self._quality_refine_worker_count(len(jobs))
        if workers <= 1:
            refined = [
                self._preserve_runtime_budget_trace(
                    seed,
                    refine_quality_candidate(inst, seed, self.config, profile),
                )
                for seed, profile in jobs
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                refined = list(
                    pool.map(
                        lambda item: self._preserve_runtime_budget_trace(
                            item[0],
                            refine_quality_candidate(
                                inst, item[0], self.config, item[1]
                            ),
                        ),
                        jobs,
                    )
                )
        return candidates + refined

    def _preserve_runtime_budget_trace(
        self, source: Placement, target: Placement
    ) -> Placement:
        trace = getattr(source, "runtime_budget_trace", None)
        if trace is not None and getattr(target, "runtime_budget_trace", None) is None:
            target.runtime_budget_trace = trace
        return target

    def _large_case_repair_profiles(self) -> list[str]:
        raw = os.environ.get("FLOORSET_LARGE_CASE_REPAIR_PROFILES", "normal")
        profiles = [part.strip() for part in raw.split(",") if part.strip()]
        allowed = {"normal", "large_boundary"}
        filtered = [profile for profile in profiles if profile in allowed]
        return filtered or ["normal"]

    def _uses_quality_portfolio(self, inst) -> bool:
        return is_quality_portfolio_case(inst)

    def _quality_profiles(self) -> list[str]:
        return enabled_quality_profiles()

    def _high_risk_repair_profiles(self) -> list[str]:
        raw = os.environ.get(
            "FLOORSET_HIGH_RISK_REPAIR_PROFILES",
            "normal",
        )
        profiles = [part.strip() for part in raw.split(",") if part.strip()]
        allowed = {
            "normal",
            "boundary_first",
            "grouping_first",
            "quality_refine",
            "large_boundary",
        }
        filtered = [profile for profile in profiles if profile in allowed]
        return filtered or ["normal"]

    def _uses_high_risk_portfolio(self, inst) -> bool:
        mode = (
            os.environ.get("FLOORSET_ENABLE_HIGH_RISK_PORTFOLIO", "auto")
            .strip()
            .lower()
        )
        if mode in {"0", "false", "off", "no"}:
            return False
        if mode in {"1", "true", "on", "yes"}:
            return self._is_high_risk_case(inst)
        return self._is_targeted_high_risk_case(inst)

    def _is_targeted_high_risk_case(self, inst) -> bool:
        budget = instance_risk_budget(inst)
        return budget.tier in {BudgetTier.MEDIUM, BudgetTier.HEAVY}

    def _is_high_risk_case(self, inst) -> bool:
        budget = instance_risk_budget(inst)
        return budget.tier is not BudgetTier.NONE

    def _is_dense_medium_risk_case(self, inst) -> bool:
        boundary_count = len(inst.boundary)
        grouping_budget = sum(
            max(0, len(members) - 1) for members in inst.cluster_groups.values()
        )
        dense_min_blocks = int(
            os.environ.get("FLOORSET_HIGH_RISK_DENSE_MIN_BLOCKS", "90")
        )
        dense_budget = int(
            os.environ.get("FLOORSET_HIGH_RISK_DENSE_CONSTRAINT_BUDGET", "50")
        )
        return (
            inst.block_count >= dense_min_blocks
            and boundary_count + grouping_budget >= dense_budget
        )

    def _uses_dense_shape_profiles(self, inst) -> bool:
        dense_max_blocks = int(
            os.environ.get("FLOORSET_HIGH_RISK_DENSE_SHAPE_MAX_BLOCKS", "109")
        )
        return (
            self._is_dense_medium_risk_case(inst)
            and inst.block_count <= dense_max_blocks
        )

    def _build_candidate(self, inst, spec: CandidateSpec) -> Placement:
        saved_guidance = inst.anchor_guidance
        if spec.disable_guidance:
            inst.anchor_guidance = None
        try:
            if spec.kind == "beam":
                graph = build_hetero_floorplan_graph(inst)
                placement = construct_beam_placement(inst, self.config, graph=graph)
            else:
                placement = construct_relative_order_placement(
                    inst, self.config, profile=spec.profile
                )
            return self._repair_candidate(inst, placement, spec)
        finally:
            if spec.disable_guidance:
                inst.anchor_guidance = saved_guidance

    def _repair_candidate(
        self, inst, placement: Placement, spec: CandidateSpec
    ) -> Placement:
        before = placement
        after = self._repair_with_profile(inst, before, spec.repair_profile)
        if spec.repair_profile == "quality_refine":
            after = self._preserve_runtime_budget_trace(
                after,
                self._quality_refine_candidate(inst, after),
            )
        after = self._preserve_runtime_budget_trace(
            after,
            refine_quality_candidate(inst, after, self.config, spec.quality_profile),
        )
        self._trace_repair(
            inst,
            before,
            after,
            {
                "name": spec.name,
                "profile": spec.profile,
                "kind": spec.kind,
                "repair_profile": spec.repair_profile,
                "quality_profile": spec.quality_profile,
            },
        )
        return after

    def _repair_with_profile(
        self, inst, placement: Placement, repair_profile: str
    ) -> Placement:
        return _repair_with_profile_worker(inst, placement, self.config, repair_profile)

    def _select_best_candidate(self, inst, candidates: list[Placement]) -> Placement:
        return min(
            candidates, key=lambda placement: self._candidate_rank(inst, placement)
        )

    def _candidate_rank(self, inst, placement: Placement) -> tuple:
        metrics = self._placement_metrics(inst, placement)
        if os.environ.get("FLOORSET_CANDIDATE_RANK_POLICY", "v10_proxy") == "soft_first":
            soft = self._soft_total(metrics)
            return (
                int(metrics["overlap_count"]),
                float(soft),
                int(metrics["boundary_violations"]),
                int(metrics["group_violations"]),
                v10_proxy_cost(inst, placement, metrics),
            )
        return v10_proxy_rank(inst, placement, metrics)

    def _placement_metrics(self, inst, placement: Placement) -> dict[str, float | int]:
        return placement_metrics(inst, placement)

    def _quality_refine_candidate(self, inst, placement: Placement) -> Placement:
        best = placement.copy()
        self._preserve_runtime_budget_trace(placement, best)
        best_metrics = self._placement_metrics(inst, best)
        best_soft = self._soft_total(best_metrics)
        max_blocks = int(os.environ.get("FLOORSET_QUALITY_REFINE_MAX_BLOCKS", "32"))
        max_slots = int(os.environ.get("FLOORSET_QUALITY_REFINE_MAX_SLOTS", "48"))
        movable = [block for block in best.rects if block not in inst.preplaced]
        movable.sort(
            key=lambda block: (
                -(
                    len(inst.b2b_by_block.get(block, []))
                    + len(inst.p2b_by_block.get(block, []))
                ),
                block,
            )
        )

        for block in movable[:max_blocks]:
            rect = best.rects[block]
            others = [other for idx, other in best.rects.items() if idx != block]
            slot_points = candidate_frontier_points(others)[:max_slots]
            for x, y in slot_points:
                candidate = Rect(max(0.0, x), max(0.0, y), rect.width, rect.height)
                if (
                    abs(candidate.x - rect.x) <= 1e-9
                    and abs(candidate.y - rect.y) <= 1e-9
                ):
                    continue
                if not first_non_overlapping(candidate, others):
                    continue
                trial = best.copy()
                trial.rects[block] = candidate
                metrics = self._placement_metrics(inst, trial)
                if (
                    int(metrics["overlap_count"]) > 0
                    or self._soft_total(metrics) > best_soft
                ):
                    continue
                if v10_proxy_better(
                    inst,
                    trial,
                    best,
                    candidate_metrics=metrics,
                    current_metrics=best_metrics,
                    allow_tie_soft=True,
                    allow_proxy_regression=False,
                ):
                    best = trial
                    best_metrics = metrics
                    best_soft = self._soft_total(metrics)
                    rect = candidate
                    others = [
                        other for idx, other in best.rects.items() if idx != block
                    ]
        return best

    def _soft_total(self, metrics: dict[str, float | int]) -> int:
        return (
            int(metrics["boundary_violations"])
            + int(metrics["group_violations"])
            + int(metrics["mib_violations"])
        )

    def _trace_repair(
        self, inst, before: Placement, after: Placement, candidate: dict[str, str]
    ) -> None:
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
        runtime_budget = getattr(after, "runtime_budget_trace", None)
        if runtime_budget is not None:
            row["runtime_budget"] = runtime_budget
        with open(trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _try_anchor_guidance(self, inst) -> Optional[AnchorGuidance]:
        checkpoint = self._resolve_checkpoint_path(
            os.environ.get(self.config.checkpoint_env)
        )
        if not checkpoint:
            return None
        try:
            from floorset_arch.features import (
                build_anchor_edge_tensors,
                build_anchor_hgt_graph_inputs,
                build_anchor_node_features,
                build_anchor_transformer_graph_inputs,
            )
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
                    encoder_type=str(payload.get("encoder_type", "mpnn")),
                    num_heads=int(payload.get("num_heads", 4)),
                    structural_feat_dim=int(payload.get("structural_feat_dim", 0)),
                    edge_type_count=int(payload.get("edge_type_count", 1)),
                    hgt_node_feat_dims=payload.get("hgt_node_feat_dims", {}),
                    hgt_relation_specs=tuple(
                        tuple(relation)
                        for relation in payload.get("hgt_relation_specs", ())
                    ),
                )
                model.load_state_dict(payload["model_state_dict"], strict=False)
                self._checkpoint_config = {
                    "node_feat_dim": int(payload["node_feat_dim"]),
                    "hidden_dim": int(payload.get("hidden_dim", 160)),
                    "layers": int(payload.get("layers", 5)),
                    "dropout": float(payload.get("dropout", 0.05)),
                    "encoder_type": str(payload.get("encoder_type", "mpnn")),
                    "num_heads": int(payload.get("num_heads", 4)),
                    "structural_feat_dim": int(payload.get("structural_feat_dim", 0)),
                    "edge_type_count": int(payload.get("edge_type_count", 1)),
                    "hgt_node_feat_dims": payload.get("hgt_node_feat_dims", {}),
                    "hgt_relation_specs": tuple(
                        tuple(relation)
                        for relation in payload.get("hgt_relation_specs", ())
                    ),
                    "has_pair_head": bool(payload.get("has_pair_head", False)),
                }
                model.eval()
                self._checkpoint_model = model
                self._checkpoint_key = checkpoint
            else:
                model = self._checkpoint_model

            with torch.no_grad():
                node_feat, scale = build_anchor_node_features(
                    inst, device=torch.device("cpu")
                )
                if node_feat.shape[1] != int(
                    self._checkpoint_config.get("node_feat_dim", node_feat.shape[1])
                ):
                    return None
                edge_type = None
                structural_feat = None
                hgt_node_features = None
                hgt_edge_index = None
                hgt_edge_attr = None
                if self._checkpoint_config.get("encoder_type") == "graph-transformer":
                    graph_inputs = build_anchor_transformer_graph_inputs(
                        inst, device=torch.device("cpu")
                    )
                    edge_index = graph_inputs.edge_index
                    edge_attr = graph_inputs.edge_attr
                    edge_type = graph_inputs.edge_type
                    structural_feat = graph_inputs.node_structural_features
                elif self._checkpoint_config.get("encoder_type") == "hgt":
                    graph_inputs = build_anchor_hgt_graph_inputs(
                        inst, device=torch.device("cpu")
                    )
                    node_feat = graph_inputs.node_features["block"]
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0, 1), dtype=torch.float32)
                    hgt_node_features = graph_inputs.node_features
                    hgt_edge_index = graph_inputs.edge_index
                    hgt_edge_attr = graph_inputs.edge_attr
                else:
                    edge_index, edge_attr = build_anchor_edge_tensors(
                        inst, device=torch.device("cpu")
                    )
                pairs = None
                if self._checkpoint_config.get("has_pair_head"):
                    pairs = torch.tensor(
                        [
                            (i, j)
                            for i in range(inst.block_count)
                            for j in range(i + 1, inst.block_count)
                        ],
                        dtype=torch.long,
                    )
                pred = model(
                    node_feat,
                    edge_index,
                    edge_attr,
                    edge_type=edge_type,
                    structural_feat=structural_feat,
                    hgt_node_features=hgt_node_features,
                    hgt_edge_index=hgt_edge_index,
                    hgt_edge_attr=hgt_edge_attr,
                    pairs=pairs,
                )
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
        anchor_only = _env_flag("FLOORSET_GUIDANCE_ANCHOR_ONLY")
        disable_pairwise = anchor_only or _env_flag(
            "FLOORSET_GUIDANCE_DISABLE_PAIRWISE"
        )
        disable_aspect = anchor_only or _env_flag("FLOORSET_GUIDANCE_DISABLE_ASPECT")
        disable_priority = anchor_only or _env_flag("FLOORSET_GUIDANCE_DISABLE_PRIORITY")
        anchors = pred["anchor"].detach().cpu() * max(float(scale), 1.0)
        priority_tensor = pred.get("priority")
        aspect_tensor = pred.get("log_aspect")
        guidance = AnchorGuidance(scale=max(float(scale), 1.0), source="anchor_gnn")
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
            if (
                not disable_aspect
                and aspect_tensor is not None
                and i not in inst.fixed
                and i not in inst.preplaced
            ):
                log_aspect = float(aspect_tensor.detach().cpu()[i])
                guidance.log_aspect[i] = log_aspect
                aspect = math.exp(max(-2.5, min(2.5, log_aspect)))
                area = max(1.0, float(inst.area_targets[i]))
                width = math.sqrt(area * aspect)
                height = math.sqrt(area / aspect)
            cx = float(anchors[i, 0])
            cy = float(anchors[i, 1])
            if math.isfinite(cx) and math.isfinite(cy):
                guidance.rect_priors[i] = Rect(
                    max(0.0, cx - width / 2.0),
                    max(0.0, cy - height / 2.0),
                    width,
                    height,
                )
            if not disable_priority and priority_tensor is not None:
                guidance.priority[i] = float(priority_tensor.detach().cpu()[i])
        pair_logits = pred.get("pair_logits")
        if not disable_pairwise and pair_logits is not None and pairs is not None:
            logits = pair_logits.detach().cpu()
            for pair, logit in zip(pairs.detach().cpu().tolist(), logits.tolist()):
                guidance.pairwise_axis[(int(pair[0]), int(pair[1]))] = (
                    float(logit[0]),
                    float(logit[1]),
                )
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
        """Pick one relative-order profile from cheap instance statistics.

        Computing both soft and compact profiles is useful for ablations.  The
        features below are cheap statistics that separate pin-heavy compact wins
        from net-heavy / boundary-sensitive soft wins without touching the
        evaluator loop.  Under no-runtime tuning, this heuristic should be
        rechecked against `total_score_no_runtime` before it blocks a heavier
        large-case candidate matrix.
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

        if p2b_count < 100 and b2b_count > 2000:
            return ["compact"]
        if p2b_count < 100:
            return ["soft"]
        if p2b_count > 2500:
            return ["soft"] if b2b_count > 5000 else ["compact"]
        if b2b_count > 5000:
            return ["soft"] if p2b_count < 200 else ["compact"]
        if cluster_count <= 3:
            if p2b_count < 700 and b2b_count < 1300 and preplaced_count <= 3:
                return ["soft"]
            return ["compact"]
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
        return v10_proxy_cost(inst, placement, placement_metrics(inst, placement))

    def _no_runtime_proxy_cost(
        self, inst, placement: Placement, metrics: dict[str, float | int]
    ) -> float:
        return v10_proxy_cost(inst, placement, metrics)


ArchitectureV4Optimizer = ArchitectureV5Optimizer
