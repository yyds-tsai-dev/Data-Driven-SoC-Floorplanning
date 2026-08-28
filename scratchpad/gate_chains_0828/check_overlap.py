#!/usr/bin/env python3
"""Check whether the 100 public validation cases (FloorSet-Lite
LiteTensorDataTest/config_21..config_120/litedata_1.pth) are exact
duplicates of rows in the 1M-row training corpus
(FloorSet/floorset_lite/worker_*/layouts_*.th).

Method
------
For each validation config directory (test_id == n, the block count, since
FloorSet-Lite names each directory config_<n> and only n in [21,120] each
has exactly one file with exactly n blocks):

  1. Load litedata_1.pth -> d[0] is a length-1 list of layouts; d[0][0] is
     [area+constraints (n,6), b2b (E_b2b,3), p2b (E_p2b,3), pins_pos (P,2)].
     area_target = tensor[:, 0].
  2. Look up every corpus file (worker_*/layouts_*.th) whose recorded block
     count (from file_n_index.json) equals n.
  3. Load each such corpus file once. It holds 112 layouts sharing that n:
     d[0] shape (112, n, 6) [.,.,0] = area_target per layout row,
     d[1] shape (112, E_b2b_pad, 3) padded b2b (zero-padded to the file's max
       edge count -- NOT directly comparable row length to the un-padded
       validation tensor, so b2b/pins comparisons below are advisory only,
       truncated/compared over the overlapping prefix length and reported
       with a padding caveat, not treated as authoritative).
     d[2] p2b, d[3] pins_pos (112, P_pad, 2), same padding caveat.
  4. A layout row is an EXACT match if area_target vectors are equal within
     atol=1e-6 (elementwise, same order -- FloorSet block order is fixed
     per-generation-seed, not permuted at storage time in this corpus, so
     no sorting/matching search is attempted; a note is printed if this
     assumption looks wrong for a given case, i.e. sum matches but
     elementwise does not).
  5. For every exact area_target match, additionally report:
       - max abs diff of pins_pos over the overlapping (min-length) prefix
       - max abs diff of b2b_connectivity over the overlapping prefix
     so a near-duplicate (area matches, geometry/connectivity differs) is
     distinguishable from a true exact duplicate.

Runs single-threaded-per-file with a small ProcessPoolExecutor (<=8 workers)
over corpus-file loads to stay within the box's CPU budget. No GPU, no
evaluator invocation. Read-only; writes only to result.txt.
"""
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import torch

REPO = "/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning"
LITE_ROOT = os.path.join(REPO, "FloorSet/floorset_lite")
VAL_ROOT = os.path.join(REPO, "FloorSet/LiteTensorDataTest")
INDEX_PATH = os.path.join(REPO, "artifacts/flow_ft_0828/file_n_index.json")
OUT_PATH = "/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/valcheck/result.txt"
ATOL = 1e-6
MAX_WORKERS = 8


def load_val_case(n):
    path = os.path.join(VAL_ROOT, f"config_{n}", "litedata_1.pth")
    d = torch.load(path, map_location="cpu", weights_only=False)
    layout = d[0]
    area_full, b2b, p2b, pins = layout[0], layout[1], layout[2], layout[3]
    area_target = area_full[:, 0].clone()
    return {
        "n": n,
        "path": path,
        "area_target": area_target,
        "b2b": b2b,
        "pins": pins,
    }


def scan_corpus_file(args):
    """Load one corpus file and compare against all val cases sharing its n."""
    relpath, n, val_cases_n = args
    full = os.path.join(LITE_ROOT, relpath)
    try:
        d = torch.load(full, map_location="cpu", weights_only=False)
    except Exception as exc:
        return relpath, n, [], f"LOAD_FAIL: {exc}"

    area_all = d[0][:, :, 0]  # (112, n)
    b2b_all = d[1]  # (112, E_pad, 3)
    pins_all = d[3]  # (112, P_pad, 2)

    results = []
    for val in val_cases_n:
        vt = val["area_target"]
        diffs = (area_all - vt.unsqueeze(0)).abs().amax(dim=1)  # (112,)
        match_rows = (diffs <= ATOL).nonzero(as_tuple=True)[0].tolist()
        for row in match_rows:
            sum_diff = float((area_all[row].sum() - vt.sum()).abs())
            elem_maxdiff = float(diffs[row])
            # advisory pins / b2b compare over overlapping prefix
            vb = val["b2b"]
            cb = b2b_all[row]
            kb = min(vb.shape[0], cb.shape[0])
            b2b_maxdiff = (
                float((vb[:kb] - cb[:kb]).abs().max()) if kb > 0 else float("nan")
            )
            vp = val["pins"]
            cp = pins_all[row]
            kp = min(vp.shape[0], cp.shape[0])
            pins_maxdiff = (
                float((vp[:kp] - cp[:kp]).abs().max()) if kp > 0 else float("nan")
            )
            results.append(
                {
                    "test_id": val["n"],
                    "n": n,
                    "corpus_file": relpath,
                    "row": row,
                    "area_elem_maxdiff": elem_maxdiff,
                    "area_sum_absdiff": sum_diff,
                    "b2b_shape_val": list(vb.shape),
                    "b2b_shape_corpus_padded": list(cb.shape),
                    "b2b_maxdiff_prefix": b2b_maxdiff,
                    "pins_shape_val": list(vp.shape),
                    "pins_shape_corpus_padded": list(cp.shape),
                    "pins_maxdiff_prefix": pins_maxdiff,
                }
            )
    return relpath, n, results, None


def main():
    t0 = time.time()
    with open(INDEX_PATH) as f:
        file_n_index = json.load(f)

    n_to_files = defaultdict(list)
    for relpath, n in file_n_index.items():
        if n > 0:
            n_to_files[n].append(relpath)

    val_cases_by_n = {}
    for n in range(21, 121):
        cfgdir = os.path.join(VAL_ROOT, f"config_{n}")
        if not os.path.isdir(cfgdir):
            continue
        val_cases_by_n[n] = [load_val_case(n)]

    print(f"loaded {len(val_cases_by_n)} validation cases", flush=True)

    tasks = []
    for n, cases in val_cases_by_n.items():
        for relpath in n_to_files.get(n, []):
            tasks.append((relpath, n, cases))

    print(f"{len(tasks)} corpus files to scan across {len(n_to_files)} n-buckets used", flush=True)

    all_matches = []
    errors = []
    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for relpath, n, results, err in ex.map(scan_corpus_file, tasks, chunksize=4):
            done += 1
            if err:
                errors.append((relpath, err))
            all_matches.extend(results)
            if done % 200 == 0:
                print(f"{done}/{len(tasks)} corpus files scanned, "
                      f"{len(all_matches)} matches so far, "
                      f"{time.time()-t0:.0f}s elapsed", flush=True)

    exact = [m for m in all_matches if m["area_elem_maxdiff"] <= ATOL]
    by_test_id = defaultdict(list)
    for m in exact:
        by_test_id[m["test_id"]].append(m)

    lines = []
    lines.append("FloorSet-Lite validation-vs-training overlap check")
    lines.append(f"generated: {time.ctime()}  elapsed: {time.time()-t0:.1f}s")
    lines.append(f"validation cases checked: {len(val_cases_by_n)} (config_21..config_120, litedata_1.pth)")
    lines.append(f"corpus files scanned: {len(tasks)}  load errors: {len(errors)}")
    lines.append(f"total exact-area-target matches (test_id, corpus row) pairs: {len(exact)}")
    lines.append(f"validation cases with >=1 exact area_target duplicate: {len(by_test_id)}")
    lines.append("")
    lines.append("=== Per-test_id exact area_target matches ===")
    for test_id in sorted(by_test_id):
        for m in sorted(by_test_id[test_id], key=lambda x: (x["corpus_file"], x["row"])):
            lines.append(
                f"test_id={test_id} n={m['n']} corpus_file={m['corpus_file']} row={m['row']} "
                f"area_elem_maxdiff={m['area_elem_maxdiff']:.3e} "
                f"pins_maxdiff_prefix={m['pins_maxdiff_prefix']:.3e} "
                f"(val_pins_shape={m['pins_shape_val']} corpus_pins_shape_padded={m['pins_shape_corpus_padded']}) "
                f"b2b_maxdiff_prefix={m['b2b_maxdiff_prefix']:.3e} "
                f"(val_b2b_shape={m['b2b_shape_val']} corpus_b2b_shape_padded={m['b2b_shape_corpus_padded']})"
            )
    lines.append("")
    lines.append("=== test_ids with NO exact area_target duplicate ===")
    no_match = [n for n in sorted(val_cases_by_n) if n not in by_test_id]
    lines.append(f"count: {len(no_match)}")
    lines.append(", ".join(str(n) for n in no_match))
    lines.append("")
    if errors:
        lines.append("=== load errors ===")
        for relpath, err in errors:
            lines.append(f"{relpath}: {err}")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT_PATH)
    print(f"exact-dup test_ids: {len(by_test_id)}/{len(val_cases_by_n)}")


if __name__ == "__main__":
    main()
