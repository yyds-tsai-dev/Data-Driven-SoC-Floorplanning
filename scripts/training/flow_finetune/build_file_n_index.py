#!/usr/bin/env python3
"""Scan FloorSet/floorset_lite and record n (block count) per layout file.

Each layouts_*.th file holds 112 layouts that all share the same block count,
and the input tensor's dim-1 IS that block count (no intra-file padding), so
the shape alone identifies n.  Output: JSON {relpath: n}.
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3] / "FloorSet" / "floorset_lite"
OUT = Path(__file__).resolve().parents[3] / "artifacts/flow_finetune_round1_0828_tailT24_300k/file_n_index.json"


def n_of(path):
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        return str(Path(path).relative_to(ROOT)), int(d[0].shape[1])
    except Exception as exc:  # pragma: no cover
        return str(Path(path).relative_to(ROOT)), -1


def main():
    files = []
    for w in range(100):
        files.extend(sorted(glob.glob(str(ROOT / f"worker_{w}" / "layouts*.th"))))
    print(f"{len(files)} files", flush=True)
    out = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, (rel, n) in enumerate(ex.map(n_of, files, chunksize=16)):
            out[rel] = n
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(files)}", flush=True)
    OUT.write_text(json.dumps(out))
    from collections import Counter
    c = Counter(out.values())
    print("n range", min(c), max(c), "distinct", len(c))
    print("files per n: min", min(c.values()), "max", max(c.values()))
    bad = [k for k, v in out.items() if v < 0]
    print("failed:", len(bad))


if __name__ == "__main__":
    main()
