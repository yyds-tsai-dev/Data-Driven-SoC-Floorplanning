"""Batch plumbing for the IC/DC engine: FloorSet tensors -> energy batch dict.

Two sources, one output contract:

  * `LiteTensorDataTest` (the 100 validation cases) -- used for the offline
    bank dump and for the energy-vs-evaluator correlation checks;
  * `floorset_lite` (the 1M training set) -- used for the data-free
    fine-tune.  Only `input_data` reaches the model; `metrics_sol` is read for
    the two normalising scalars and never enters a model input.

The training set stores one block count per file (verified over all 9000
files), so a band-restricted sampler is just a file filter -- no per-sample
scan, no shuffling cost.
"""

from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

REPO = Path(__file__).resolve().parents[2]
LITE_ROOT = REPO / "FloorSet" / "floorset_lite"


# ---------------------------------------------------------------------------
def poly_to_bbox(block: torch.Tensor):
    valid = block[block[:, 0] != -1]
    if len(valid) == 0:
        return (0.0, 0.0, 1.0, 1.0)
    lo = valid.min(dim=0).values
    hi = valid.max(dim=0).values
    return (float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1]))


def target_positions_from_rects(rects, cons, n: int) -> torch.Tensor:
    """Fixed/preplaced geometry -- the part of the golden layout that is a
    contest *input*, matching `known_target_positions_from_fp`."""
    tp = torch.full((n, 4), -1.0, dtype=torch.float64)
    for i in range(n):
        fixed = bool(cons[i, 0] != 0) if cons.shape[1] > 0 else False
        pre = bool(cons[i, 1] != 0) if cons.shape[1] > 1 else False
        if fixed or pre:
            tp[i, 2] = rects[i][2]
            tp[i, 3] = rects[i][3]
        if pre:
            tp[i, 0] = rects[i][0]
            tp[i, 1] = rects[i][1]
    return tp


def load_test_cases(evaluator, data_path: Optional[str] = None) -> List[Dict]:
    """The 100 validation instances, with evaluator baselines attached."""
    root = str(data_path or (REPO / "FloorSet"))
    ds = evaluator.FloorplanDatasetLiteTest(root)
    out = []
    for idx in range(len(ds)):
        s = ds[idx]
        area_target, b2b, p2b, pins, cons = s["input"]
        polygons, metrics = s["label"]
        n = int((area_target != -1).sum().item())
        rects = [poly_to_bbox(polygons[i]) for i in range(n)]
        hb = evaluator.calculate_hpwl_b2b(rects, b2b)
        hp = evaluator.calculate_hpwl_p2b(rects, p2b, pins)
        area = evaluator.calculate_bbox_area(rects)
        if metrics is not None and len(metrics) >= 8:
            if metrics[0] > 0:
                area = float(metrics[0])
            if metrics[-2] > 0:
                hb = float(metrics[-2])
            if metrics[-1] >= 0:
                hp = float(metrics[-1])
        out.append({
            "test_id": idx, "n": n, "rects": rects,
            "area": area_target[:n].to(torch.float64).reshape(n),
            "cons": cons[:n].to(torch.float64),
            "b2b": b2b.to(torch.float64), "p2b": p2b.to(torch.float64),
            "pins": pins.to(torch.float64),
            "tp": target_positions_from_rects(rects, cons[:n], n),
            "hpwl_ref": float(hb + hp), "area_ref": float(area),
            "golden": rects,
        })
    return out


# ---------------------------------------------------------------------------
def collate(cases: Sequence[Dict], device="cpu",
            dtype=torch.float32) -> Dict[str, torch.Tensor]:
    """Pad a list of per-case dicts into the batch contract used everywhere.

    Padding uses ``area = -1`` (the FloorSet convention, which
    `valid_block_mask` already keys off) and ``-1`` for edge/pin rows, which
    every consumer masks out.
    """
    from .energy import n_soft as _n_soft

    B = len(cases)
    N = max(c["n"] for c in cases)
    E = max(int(c["b2b"].shape[0]) for c in cases)
    Ep = max(int(c["p2b"].shape[0]) for c in cases)
    P = max(int(c["pins"].shape[0]) for c in cases)
    NC = max(int(c["cons"].shape[1]) for c in cases)

    area = torch.full((B, N), -1.0, dtype=dtype)
    cons = torch.zeros((B, N, NC), dtype=dtype)
    tp = torch.full((B, N, 4), -1.0, dtype=dtype)
    b2b = torch.full((B, E, 3), -1.0, dtype=dtype)
    p2b = torch.full((B, Ep, 3), -1.0, dtype=dtype)
    pins = torch.full((B, P, 2), -1.0, dtype=dtype)
    hpwl_ref = torch.zeros(B, dtype=dtype)
    area_ref = torch.zeros(B, dtype=dtype)
    for k, c in enumerate(cases):
        n = c["n"]
        area[k, :n] = c["area"].to(dtype)
        cons[k, :n, :c["cons"].shape[1]] = c["cons"].to(dtype)
        tp[k, :n] = c["tp"].to(dtype)
        b2b[k, :c["b2b"].shape[0]] = c["b2b"].to(dtype)
        p2b[k, :c["p2b"].shape[0]] = c["p2b"].to(dtype)
        pins[k, :c["pins"].shape[0]] = c["pins"].to(dtype)
        hpwl_ref[k] = c["hpwl_ref"]
        area_ref[k] = c["area_ref"]

    mask = area > 0
    scale = torch.sqrt(torch.where(mask, area, torch.zeros_like(area))
                       .sum(dim=1).clamp_min(1.0))
    batch = {
        "area": area, "cons": cons, "tp": tp, "b2b": b2b, "p2b": p2b,
        "pins": pins, "hpwl_ref": hpwl_ref.clamp_min(1e-9),
        "area_ref": area_ref.clamp_min(1e-9), "scale": scale,
        # tau_sharp: the "is this a violation" width.  The evaluator's boundary
        # epsilon is 1e-6 absolute and its grouping check is bit-exact, so this
        # wants to be as small as the working dtype allows -- 1e-5*scale is
        # ~2e-3 units, comfortably above float32's ~6e-5 rounding at coordinate
        # magnitude 600 and still 3 orders below a block dimension.
        # tau_soft: the shaping width, one block-diameter-ish, for long range.
        "tau_sharp": 1e-5 * scale, "tau_soft": 1e-2 * scale,
        "n": torch.tensor([c["n"] for c in cases], dtype=torch.long),
    }
    batch["n_soft"] = _n_soft(cons, area)
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# training-set sampler restricted to a block-count band
# ---------------------------------------------------------------------------
class BandFileSampler:
    """Draw training instances from files whose block count is in `band`.

    Files hold 112 layouts each and one block count per file, so a stratified
    draw over block counts is a draw over files.  Loading is deliberately
    file-at-a-time (the on-disk layout is one torch file per 112 layouts).
    """

    def __init__(self, band=(95, 120), index_path: Optional[str] = None,
                 seed: int = 0, workers=range(100)):
        self.rng = random.Random(seed)
        self.files_by_n: Dict[int, List[str]] = {}
        if index_path and os.path.exists(index_path):
            for path, n, _cnt in json.load(open(index_path)):
                if n > 0 and band[0] <= n <= band[1]:
                    self.files_by_n.setdefault(n, []).append(path)
        else:
            for w in workers:
                for path in glob.glob(str(LITE_ROOT / f"worker_{w}" / "layouts*")):
                    self.files_by_n.setdefault(-1, []).append(path)
        self.ns = sorted(self.files_by_n)
        self._cache_path = None
        self._cache = None

    def _load(self, path):
        if path != self._cache_path:
            self._cache = torch.load(path, map_location="cpu")
            self._cache_path = path
        return self._cache

    def draw(self, count: int, per_file: int = 8) -> List[Dict]:
        """`count` instances, stratified over block counts.

        `per_file` layouts are taken from each file visited: one torch.load is
        ~0.3 s, so drawing a few thousand one-at-a-time would spend a quarter
        of an hour in I/O for no extra diversity worth having (the pool is
        re-sampled every step anyway).
        """
        out = []
        while len(out) < count:
            n = self.rng.choice(self.ns)
            path = self.rng.choice(self.files_by_n[n])
            d = self._load(path)
            take = min(per_file, count - len(out), len(d[0]))
            for k in self.rng.sample(range(len(d[0])), take):
                out.append(self._instance(d, k))
        return out

    @staticmethod
    def _instance(d, k) -> Dict:
        row = d[0][k]
        area_all = row[:, 0]
        cons_all = row[:, 1:]
        n = int((area_all != -1).sum().item())
        fp = d[5][k][:n]                       # [w, h, x, y]
        rects = [(float(r[2]), float(r[3]), float(r[0]), float(r[1]))
                 for r in fp]
        m = d[6][k]
        return {
            "n": n, "area": area_all[:n].to(torch.float64),
            "cons": cons_all[:n].to(torch.float64),
            "b2b": d[1][k].to(torch.float64), "p2b": d[2][k].to(torch.float64),
            "pins": d[3][k].to(torch.float64),
            "tp": target_positions_from_rects(rects, cons_all[:n], n),
            "hpwl_ref": float(m[-2]) + float(m[-1]),
            "area_ref": float(m[0]),
            "golden": rects,
        }
