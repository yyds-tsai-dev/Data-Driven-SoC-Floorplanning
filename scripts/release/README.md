# Release tooling (final sprint, 2026-08-27 .. 08-31)

| Script | Purpose |
| --- | --- |
| `../pack_cadc1013.sh <out_dir>` | Build `cadc1013.tar.gz` from `src/solver` + `src/shipping` (import closure of `contest_optimizer.py`, shipped as `op_src.py`). |
| `apply_pack_variant.sh [ft2] [s16]` | Rewrite `src/shipping/op_wrapper.py` + the pack script for a shipping variant (flow checkpoint, `FLOW_SLOTS`/`NREF`) before packing. No args = 0828b settings; `ft2 s16` = B'. |
| `rehearse_package.sh [work_dir]` | Pack, extract, build a clean venv from the package's `requirements.txt`, run the official evaluator on the extracted `op_wrapper.py`, print selfcheck / legal-guard / score summary. |
| `runtime_aware_total.py` | Runtime-aware totals from result JSONs under field-speed scenarios (`--M`, `--D`). |
| `runtime_aware_pairs.py` | Paired base-vs-candidate comparison, raw and runtime-aware. |

Uploaded packages live under `submission/` (git-ignored): `FINAL_UPLOAD/cadc1013.tar.gz`
(B', md5 3f2cda42) and `FINAL_UPLOAD_C/cadc1013.tar.gz` (C', md5 6d0ca94e). Repacking changes
the tarball md5 (mtimes); compare content with an extracted `diff -r`, not the md5.
