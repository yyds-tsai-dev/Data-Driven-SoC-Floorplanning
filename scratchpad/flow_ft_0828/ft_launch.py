#!/usr/bin/env python3
"""Fine-tune launcher for the flow-matching prior (v1 -> tail-tilted continuation).

This file lives entirely under artifacts/ and changes NO shipped trainer code.
It reuses ``flow_matching_train.main`` (and therefore ``direct_diffusion_train``'s
lifecycle: EMA, checkpointing, signals, throttling) and patches exactly two
module attributes before handing over:

  * ``direct_diffusion_train.FloorplanDatasetLite``
        -> ``TrainSplitLite``: restricts the corpus to ``worker_0 .. worker_<MAX>``.
           The shadow suites (v1/v3/v5/v6) are a *late-worker holdout*: every one
           of their manifests records ``workers: [90..99]`` and every selected
           case has a ``source_file`` under ``worker_90..worker_99``.  Dropping
           those ten workers removes the entire shadow candidate pool
           (100,800 of 1,008,000 rows), leaving 907,200 training rows.
           Public validation lives in a different container
           (``LiteTensorDataTest``) and is never reachable from this dataset.

  * ``direct_diffusion_train.FileShuffleSampler``
        -> ``TailTiltSampler``: file-granular sampling as before, but files are
           drawn *with replacement* proportional to ``exp((n - 120) / TEMP)``.
           TEMP = 24 is the square root of the official case weight
           ``exp((n - n_max) / 12)`` -- a tempered importance tilt toward the
           block counts that carry the score mass, without collapsing support
           on small n.

Knobs (env, so the forwarded CLI stays a verbatim flow-trainer command line):
    FT_MAX_WORKER   highest worker index used for training      (default 89)
    FT_TAIL_TEMP    tilt temperature; <=0 disables the tilt     (default 24)
    FT_N_INDEX      path to the {relpath: n} JSON index
Everything else is passed straight through to ``flow_matching_train``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import torch

REPO = Path("/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning")
for _p in (REPO / "partner", REPO / "FloorSet" / "iccad2026contest", REPO / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import direct_diffusion_train as V1  # noqa: E402
import flow_matching_train as FM  # noqa: E402

MAX_WORKER = int(os.environ.get("FT_MAX_WORKER", "89"))
TAIL_TEMP = float(os.environ.get("FT_TAIL_TEMP", "24"))
N_INDEX = Path(os.environ.get(
    "FT_N_INDEX", str(REPO / "artifacts/flow_ft_0828/file_n_index.json")))

_WORKER_RE = re.compile(r"worker_(\d+)")
_ACTIVE = {}


class TrainSplitLite(V1.FloorplanDatasetLite):
    """FloorSet-Lite restricted to the training workers (shadow pool removed)."""

    def __init__(self, root):
        super().__init__(root)
        kept = []
        for path in self.all_files:
            m = _WORKER_RE.search(str(path))
            if m is not None and int(m.group(1)) <= MAX_WORKER:
                kept.append(path)
        dropped = len(self.all_files) - len(kept)
        self.all_files = kept
        _ACTIVE["dataset"] = self
        print(f"[ft] train split: {len(kept)} files kept, {dropped} dropped "
              f"(worker_0..worker_{MAX_WORKER}); "
              f"{len(kept) * self.layouts_per_file} rows", flush=True)


class TailTiltSampler(torch.utils.data.Sampler):
    """File-granular sampler that oversamples large-n files.

    Signature matches ``FileShuffleSampler(n_files, per_file, seed)`` so it is a
    drop-in inside ``direct_diffusion_train.main``.  One "epoch" draws
    ``n_files`` file slots *with replacement* from the tilt distribution and
    emits a random permutation of each file's rows, so batches stay
    n-homogeneous and file-cache friendly exactly as before.
    """

    def __init__(self, n_files: int, per_file: int, seed: int):
        self.n_files = n_files
        self.per_file = per_file
        self.seed = seed
        self.epoch = 0
        dataset = _ACTIVE.get("dataset")
        if dataset is None:
            raise RuntimeError("TailTiltSampler requires TrainSplitLite to be built first")
        index = json.loads(N_INDEX.read_text())
        root = REPO / "FloorSet" / "floorset_lite"
        ns = []
        for path in dataset.all_files:
            rel = str(Path(path).resolve().relative_to(root))
            n = index.get(rel, -1)
            if n <= 0:
                raise RuntimeError(f"missing n for {rel}")
            ns.append(n)
        self.ns = torch.tensor(ns, dtype=torch.float64)
        if TAIL_TEMP > 0:
            logits = (self.ns - 120.0) / TAIL_TEMP
            w = torch.exp(logits - logits.max())
        else:
            w = torch.ones_like(self.ns)
        self.weights = (w / w.sum()).to(torch.float64)
        top = float(self.weights[self.ns >= 109].sum())
        mid = float(self.weights[self.ns >= 97].sum())
        low = float(self.weights[self.ns <= 60].sum())
        print(f"[ft] tail tilt temp={TAIL_TEMP}: P(n>=109)={top:.3f} "
              f"P(n>=97)={mid:.3f} P(n<=60)={low:.3f} over {n_files} files",
              flush=True)

    def __len__(self):
        return self.n_files * self.per_file

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        files = torch.multinomial(self.weights, self.n_files, replacement=True,
                                  generator=g).tolist()
        for f in files:
            base = f * self.per_file
            for k in torch.randperm(self.per_file, generator=g).tolist():
                yield base + k


def main():
    V1.FloorplanDatasetLite = TrainSplitLite
    V1.FileShuffleSampler = TailTiltSampler
    FM.main()


if __name__ == "__main__":
    main()
