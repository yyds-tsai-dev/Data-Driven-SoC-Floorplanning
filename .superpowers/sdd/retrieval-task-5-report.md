# Retrieval Task 5 report

## Scope

- Owned changes only: `partner/my_opt_claude.py`,
  `tests/test_partner_retrieval_integration.py`, and
  `scripts/probes/run_retrieval_gate.sh`, plus this report.
- Preserved unrelated dirty `docs/official/` deletions/untracked test data.
- Did not run the full-100 gate or build, overwrite, or load a production
  retrieval index during this task.

## TDD evidence

- **RED:** `uv run pytest tests/test_partner_retrieval_integration.py -q`
  initially failed because `_select_ranked_source_quota` did not exist.
- **GREEN:** the integration file passed `8 passed` after the minimal
  opt-in loader, retrieval sampler, shared portfolio ranking, and quota
  selection implementation.
- Focused retrieval suite:
  `58 passed` with
  `uv run pytest tests/test_partner_retrieval_features.py tests/test_partner_retrieval_index.py tests/test_partner_retrieval_transfer.py tests/test_partner_retrieval_scripts.py tests/test_partner_retrieval_integration.py -q`.

## Verification evidence

- Full suite: `447 passed, 1 skipped` from `uv run pytest`.
- Evaluator interface: `PASSED`; sample runtime `0.578s` from
  `cd FloorSet/iccad2026contest && uv run python iccad2026_evaluate.py --validate ../../partner/my_opt_claude.py`.
- `uv run python -m py_compile partner/my_opt_claude.py` and `git diff --check` passed.

## Runtime, parity, and leakage audit

- Retrieval is opt-in only: empty index path, zero slots, invalid load,
  unavailable pool, and query/transfer failures retain the Direct path.
- Disabled pool sampling still calls the reviewed Direct wrapper, preserving
  raw generation, oversampling, penalty calculation, rank call, and order.
- Enabled retrieval uses the existing Direct raw batch, at most two retrieval
  transfers, one shared target-based rank, then fixed source quotas; it never
  changes `n_ref`, worker count, GPU cap, or deadline.
- The optimizer does not load FloorSet data or labels. `RetrievalIndex.load`
  remains responsible for train-only provenance/hash validation.
- Gate runner references only the existing `artifacts/retrieval/pilot64`, uses
  two retrieval slots, and sets `PARTNER_DIRECT_MIN=2.5` for both paired runs.
