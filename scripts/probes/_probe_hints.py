"""Hint providers for gen_decoder_probe: Anchor-GNN (E2), v11 diffusion (E3),
and the production column-backbone layout (D1 decoder-as-polish probe).

Replicates the model-inference call sequences from
src/floorset_arch/optimizer.py (_try_anchor_guidance / _try_diffusion_prior)
WITHOUT importing optimizer.py or refine.api (mid-edit hazard). Imports only
parser, features, nn.model, diffusion.{graph_inputs,sampling,concretize},
training.checkpoint -- all safe. The production provider additionally does a
lazy import of legalizer.column_backbone (the golden/gnn/diffusion paths never
touch it).

A "hint" is a per-block (x, y, w, h) tuple (lower-left corner). Providers
return either one hint list (GNN/production) or a list-of-hint-lists
(diffusion, one per sample) that the probe decodes and best-selects.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import torch

from floorset_arch.parser import parse_instance

Rect = Tuple[float, float, float, float]


def _inst_from_sample(sample, n: int):
    at, b2b, p2b, pins, cons = sample["input"]
    # Build opt_target_positions (preplaced xywh + fixed wh) exactly as the
    # evaluator hands them to solve() -- parse_instance reads target_rects
    # from this for preplaced/fixed handling.
    from gen_decoder_probe import _golden_rects, _opt_target_positions
    golden = _golden_rects(sample, n)
    tpos = _opt_target_positions(sample, n, golden)
    return parse_instance(n, at, b2b, p2b, pins, cons, tpos)


# =============================================================================
# E2: Anchor-GNN
# =============================================================================
class GnnHintProvider:
    def __init__(self, checkpoint: str):
        from floorset_arch.nn.model import FloorplanGNN
        from floorset_arch.training import checkpoint as checkpoint_io

        payload = checkpoint_io.load_checkpoint(checkpoint, map_location="cpu")
        self.cfg = {
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
                tuple(r) for r in payload.get("hgt_relation_specs", ())
            ),
            "hgt_relation_gate_min": float(payload.get("hgt_relation_gate_min", 0.10)),
            "pair_head_version": int(payload.get("pair_head_version", 1)),
        }
        model = FloorplanGNN(
            node_feat_dim=self.cfg["node_feat_dim"],
            hidden_dim=self.cfg["hidden_dim"],
            num_layers=self.cfg["layers"],
            dropout=self.cfg["dropout"],
            encoder_type=self.cfg["encoder_type"],
            num_heads=self.cfg["num_heads"],
            structural_feat_dim=self.cfg["structural_feat_dim"],
            edge_type_count=self.cfg["edge_type_count"],
            hgt_node_feat_dims=self.cfg["hgt_node_feat_dims"],
            hgt_relation_specs=self.cfg["hgt_relation_specs"],
            hgt_relation_gate_min=self.cfg["hgt_relation_gate_min"],
            pair_head_version=self.cfg["pair_head_version"],
        )
        model.load_state_dict(payload["model_state_dict"], strict=False)
        model.eval()
        self.model = model

    def _run(self, sample, n: int, want_pairs: bool):
        """Single forward pass. Returns (pred_dict, scale, pairs_or_None).

        When ``want_pairs`` and the checkpoint carries a v2 pair head, all
        i<j pairs (in the exact ``sample_pairs`` enumeration order used by
        training) are scored and ``pred['pair_axis4_logits']`` is populated.
        """
        from floorset_arch.features import (
            build_anchor_edge_tensors,
            build_anchor_hgt_graph_inputs,
            build_anchor_node_features,
            build_anchor_transformer_graph_inputs,
        )
        inst = _inst_from_sample(sample, n)
        dev = torch.device("cpu")
        pairs = None
        if want_pairs and self.cfg.get("pair_head_version", 1) == 2:
            pairs = torch.tensor(
                [(i, j) for i in range(n) for j in range(i + 1, n)],
                dtype=torch.long,
            )
        with torch.no_grad():
            node_feat, scale = build_anchor_node_features(inst, device=dev)
            edge_type = structural_feat = None
            hgt_nf = hgt_ei = hgt_ea = None
            enc = self.cfg["encoder_type"]
            if enc == "graph-transformer":
                gi = build_anchor_transformer_graph_inputs(inst, device=dev)
                edge_index, edge_attr = gi.edge_index, gi.edge_attr
                edge_type = gi.edge_type
                structural_feat = gi.node_structural_features
            elif enc == "hgt":
                gi = build_anchor_hgt_graph_inputs(inst, device=dev)
                node_feat = gi.node_features["block"]
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float32)
                hgt_nf, hgt_ei, hgt_ea = gi.node_features, gi.edge_index, gi.edge_attr
            else:
                edge_index, edge_attr = build_anchor_edge_tensors(inst, device=dev)
            pred = self.model(
                node_feat, edge_index, edge_attr,
                edge_type=edge_type, structural_feat=structural_feat,
                hgt_node_features=hgt_nf, hgt_edge_index=hgt_ei,
                hgt_edge_attr=hgt_ea, pairs=pairs,
            )
        return pred, scale, pairs, inst

    def _rects_from_pred(self, pred, scale, inst, n: int) -> List[Rect]:
        anchors = pred["anchor"].detach().cpu() * max(float(scale), 1.0)
        log_aspect = pred.get("log_aspect")
        out: List[Rect] = []
        for i in range(n):
            area = max(1.0, float(inst.area_targets[i]))
            if log_aspect is not None:
                la = max(-2.5, min(2.5, float(log_aspect.detach().cpu()[i])))
            else:
                la = 0.0
            e = math.exp(la)
            w = math.sqrt(area * e)
            h = math.sqrt(area / e)
            cx = float(anchors[i, 0])
            cy = float(anchors[i, 1])
            if not (math.isfinite(cx) and math.isfinite(cy)):
                cx, cy = 0.0, 0.0
            # Hint stored as (x, y, w, h) with x,y = lower-left so it matches the
            # golden-hint format the decoder consumes (it recomputes centroids).
            out.append((cx - w / 2.0, cy - h / 2.0, w, h))
        return out

    def hints(self, sample, n: int) -> List[Rect]:
        pred, scale, _pairs, inst = self._run(sample, n, want_pairs=False)
        return self._rects_from_pred(pred, scale, inst, n)

    def hints_and_pairs(self, sample, n: int):
        """Return (hint_rects, pair_map) where pair_map is a dict keyed by the
        ordered index pair (i, j) with i<j -> (pred_class:int, conf:float),
        or None if this checkpoint has no v2 pair head.

        Class semantics match losses.PAIR_AXIS4_CLASSES / build_order_dags:
          0 = i left-of j   1 = j left-of i
          2 = i below j     3 = j below i
        """
        pred, scale, pairs, inst = self._run(sample, n, want_pairs=True)
        rects = self._rects_from_pred(pred, scale, inst, n)
        if pairs is None or "pair_axis4_logits" not in pred:
            return rects, None
        logits = pred["pair_axis4_logits"].detach().cpu()
        sm = torch.softmax(logits, dim=1)
        conf, cls = sm.max(dim=1)
        pair_map: dict = {}
        pl = pairs.tolist()
        for k, (i, j) in enumerate(pl):
            pair_map[(int(i), int(j))] = (int(cls[k]), float(conf[k]))
        return rects, pair_map


# =============================================================================
# D1: production column-backbone layout as hints (decoder-as-polish probe)
# =============================================================================
class ProductionHintProvider:
    """Feed the PRODUCTION layout back to the decoder as hints.

    D1 probe: the decoder re-derives coordinates from the layout's own
    pairwise order (longest-path re-compaction, de-quantizing the column
    representation) and boundary-snaps residual wall violations. The driver
    reports paired baseline/decoded/portfolio(min) totals -- portfolio(min)
    is the deployment semantics (strict-better gate).

    Env parity with the promoted production path (.env defaults) is applied
    via setdefault so a bare run matches scripts/eval_total.sh behavior.
    Layouts are cached to ``cache_path`` (JSON keyed by case idx) so a retest
    (e.g. after wiring repair_pin_order) reuses identical layouts instead of
    re-running the ~5s/case SA.
    """

    _ENV_PARITY = (
        ("FLOORSET_COLUMN_BACKBONE", "1"),
        ("FLOORSET_SLACK_REFINE", "1"),
        ("FLOORSET_SLACK_REFINE_ASPECT", "1"),
        ("FLOORSET_SLACK_REFINE_VSNAP", "1"),
        ("FLOORSET_FAST_EVAL", "1"),
        ("FLOORSET_WINDOW_REPACK", "1"),
    )

    def __init__(self, cache_path: Optional[str] = None):
        import json

        for k, v in self._ENV_PARITY:
            os.environ.setdefault(k, v)
        self.cache_path = cache_path
        self._cache: dict = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = {int(k): v for k, v in json.load(f).items()}

    def _save(self) -> None:
        import json

        if not self.cache_path:
            return
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({str(k): v for k, v in self._cache.items()}, f)
        os.replace(tmp, self.cache_path)

    def hints(self, sample, n: int, idx: Optional[int] = None) -> List[Rect]:
        if idx is not None and idx in self._cache:
            return [tuple(r) for r in self._cache[idx]]
        # Lazy import: keeps the golden/gnn/diffusion probe paths free of any
        # column_backbone dependency (see module docstring).
        from floorset_arch.legalizer.column_backbone import (
            solve_with_column_backbone,
        )
        from gen_decoder_probe import _golden_rects, _opt_target_positions

        at, b2b, p2b, pins, cons = sample["input"]
        golden = _golden_rects(sample, n)
        tpos = _opt_target_positions(sample, n, golden)
        rects = solve_with_column_backbone(n, at, b2b, p2b, pins, cons, tpos)
        out = [tuple(float(v) for v in r) for r in rects[:n]]
        if idx is not None:
            self._cache[idx] = [list(r) for r in out]
            self._save()
        return out


# =============================================================================
# E3: v11 diffusion
# =============================================================================
class DiffusionHintProvider:
    def __init__(self, checkpoint: str, use_ema: bool = True,
                 steps: int = 0, n_samples: int = 1):
        self.checkpoint = checkpoint
        self.use_ema = use_ema
        self.steps = steps
        self.n_samples = max(1, n_samples)
        self._model = None
        self._graph_cache = None

    def hints(self, sample, n: int) -> List[List[Rect]]:
        from floorset_arch.diffusion.graph_inputs import build_diffusion_graph_inputs
        from floorset_arch.diffusion.sampling import (
            load_diffusion_checkpoint,
            sample_diffusion_prior,
        )
        from floorset_arch.diffusion.concretize import (
            concretize_diffusion_prior,
            placement_from_tensor_candidate,
        )
        dev = torch.device("cpu")
        inst = _inst_from_sample(sample, n)
        graph_inputs = build_diffusion_graph_inputs(inst, device=dev)
        if self._model is None:
            self._model = load_diffusion_checkpoint(
                self.checkpoint, graph_inputs, map_location="cpu",
                use_ema=self.use_ema,
            )
        steps = self.steps if self.steps > 0 else 64
        prior = sample_diffusion_prior(
            self._model, graph_inputs,
            samples=self.n_samples, steps=steps, seed=0,
        )
        batch = concretize_diffusion_prior(inst, prior)
        n_cand = batch.rect_xywh.shape[0]
        hint_lists: List[List[Rect]] = []
        for idx in range(n_cand):
            pl = placement_from_tensor_candidate(inst, batch, idx)
            hint_lists.append([pl.rects[i].as_tuple() for i in range(n)])
        return hint_lists
