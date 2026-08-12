# 2026-08-12 grouping bridge package verification: PROMOTE

## Decision

The grouping-bridge package is **PROMOTE/PASS**. Fresh extraction passed all
interface, hard-feasibility, weighted no-runtime, and runtime gates. The prior
`submission/cadc1013_0811d_tagcompress.tar.gz` fallback remains untouched.

## Evidence

Source HEAD: `b094643a6debe0a3a825039cb6052b4a82b7281d`. Amended G0/G1 was
independently clean-approved; its literal online ON no-runtime baseline is
`1.145723926201144` (with the documented deadline-SA noise caveat).

Reviewed partner sources were copied byte-for-byte. The wrapper enables
`PARTNER_TAG_COMPRESS=1` and `PARTNER_GROUP_BRIDGE=1`; no group debug or group
budget override was added (public budget default remains `0.02`).

Focused tests: `uv run pytest tests/test_partner_group_bridge.py
tests/test_partner_tag_compress.py -q` -> **27 passed**.

Archive `submission/cadc1013_0812_groupbridge.tar.gz`: 32 entries, exactly two
internal checkpoints, no cache/compiled-kernel/log/artifact/scratchpad entries.
MD5 `a7803d1c9e0c47dd6ca9649a98aab29c`; SHA256
`e1ca2f4be4e5bdafa7cd857bb9ae7e7a121f68b1a603a50516fb78209d14136b`.
Fallback `0811d` MD5 `aab0c6553e387275301674095a0cd709` unchanged.

Fresh extraction: `/tmp/gbridge-package-14UDW0`; source closure asserted with
`cmp`. Validation passed (exit 0, sample runtime 0.057 s). Full-100 passed
evaluator execution (exit 0) and wrote
`artifacts/partner_eval/gbridge_package_full100.json`.

## Threshold matrix

| Gate | Result | Threshold | Status |
|---|---:|---:|---|
| validation | exit 0 | exit 0 | PASS |
| records | 100 | 100 | PASS |
| feasible | 100/100 | 100/100 | PASS |
| average runtime | 0.29881621031556277 s | <= 0.300 s | PASS |
| no-runtime score (`total_score_no_runtime`) | 1.1437448258795715 | <= 1.145723926201144 | PASS |
| evaluator errors | 0 | 0 | PASS |

Logs: `.superpowers/sdd/gbridge-package-validate.log` and
`.superpowers/sdd/gbridge-package-full100.log`.

## Adjudication note

The required gate is the evaluator's top-level weighted
`total_score_no_runtime=1.1437448258795715`, passing by `0.0019791003215725`.
The informational unweighted case average
`summary.avg_cost_no_runtime=1.3052033628311677` is not the promotion metric.
