# Task 4 source-fix A RED2

Commit: `2f844dc` (`test: complete topology teacher evidence regressions`).

Updated only `tests/test_icdc_topology_prior.py`. The evidence-A probes now
cover CLI help, padded connectivity retention, malformed source transactions,
exact/boolean preflight validation, outcome numeric rejection, and canonical
Infinity rejection. Existing unrelated scratchpad files were preserved.

Verification: `git diff --check` passed. The targeted selection collected 20
tests; after correcting the fixture dtype, the source/evidence assertions are
intended as RED probes against the current production implementation. A full
file run was not performed in this worker turn.
