# Fast Final Grouping Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evidence-gate the approved default-off fast grouping bridge, then promote it only if it improves the evaluator-facing solver while retaining 100/100 hard feasibility and average runtime at or below 0.300 seconds.

**Architecture:** A narrow public wrapper in `partner/violation_killer.py` reuses the existing deterministic grouping candidate generator and exact hard guards under a soft per-case deadline. The existing final `_tag_compress` hook in `partner/contest_optimizer.py` becomes a two-pass final hook: tag compression first, grouping bridge second, with one shared scorer and per-stage failure containment. `submission/cadc1013/` remains untouched until the saved-layout G0 and three-pair online G1 gates both pass.

**Tech Stack:** Python 3.12, NumPy, PyTorch fixtures, pytest, official ICCAD evaluator, `uv`.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-11-fast-grouping-bridge-design.md`; do not reopen its product intent.
- `PARTNER_GROUP_BRIDGE` remains default off through G0/G1. With both final-pass flags off, return the exact same input object before imports or scorer construction.
- Production code may not reference validation case IDs, saved layouts, golden positions, or evaluator results.
- The bridge is deterministic and consumes no random numbers.
- Reject invalid, non-finite, zero, or negative budgets by returning the exact input object; `0.02` seconds is a soft deadline.
- A committed bridge must satisfy all three strict conditions: grouping violations decrease, evaluator-faithful total soft violations decrease, and `_Ctx` proxy score improves.
- A committed bridge must pass `_final_guards_ok`: no overlap, preplaced rectangles byte-identical, fixed/preplaced dimensions unchanged, and soft-block area within the existing hard tolerance.
- Tag compression runs before grouping; a grouping failure must preserve an already accepted tag-compression result.
- Construct or reuse at most one `_ColumnOptimizer` scorer when either or both final passes are enabled.
- Do not modify `submission/cadc1013/` before G1 passes. The verified `submission/cadc1013_0811d_tagcompress.tar.gz` remains the fallback.
- G0 requires 100/100 hard feasibility, official weighted no-runtime gain at least `0.003`, and bridge computation at most `0.003` seconds per case on average.
- G1 requires three reversed-order full100 ON/OFF pairs: ON improves no-runtime in all three, mean delta is at most `-0.002`, every ON average runtime is at most `0.300` seconds, and mean paired runtime increase is at most `0.003` seconds.
- Implementation follows red-green-refactor; the implementer must record the failing and passing commands/output.

---

### Task 1: Grouping-only public operation

**Files:**
- Create: `tests/test_partner_group_bridge.py`
- Modify: `partner/violation_killer.py`

**Interfaces:**
- Consumes: existing `_grouping_count`, `_violations_exact`, `_Ctx`, `_fix_grouping`, and `_final_guards_ok` in `partner/violation_killer.py`.
- Produces: `bridge_grouping_violations(opt, out, budget_s=0.02) -> List[Rect]`.

- [ ] **Step 1: Write the failing direct-wrapper tests**

Create fixtures using two or three unit rectangles, `torch.ones(n)` area targets, a `(n, 5)` constraint tensor with cluster IDs in column 3, `torch.full((n, 4), -1.0)` target positions, and empty `(0, 3)` b2b/p2b plus `(0, 2)` pins. Construct the real scorer with:

```python
def _scorer(rects, areas, constraints, targets, b2b, p2b, pins):
    return csl._ColumnOptimizer(
        rects, areas, constraints, targets, b2b, p2b, pins,
        time.time() + 60.0, seed=0,
    )
```

Add focused tests named `test_invalid_shape_and_nonpositive_or_nonfinite_budget_are_identity`, `test_no_cluster_and_connected_cluster_are_identity`, `test_disconnected_movable_group_is_bridged_and_strictly_improves`, `test_preplaced_component_is_never_moved`, `test_proxy_non_improvement_rolls_back`, `test_grouping_non_improvement_rolls_back`, `test_final_guard_failure_rolls_back`, `test_exception_is_contained`, and `test_bridge_is_deterministic`.

For every rollback/identity test use `got is out`. For accepted layouts assert `_grouping_count(opt, Q) < _grouping_count(opt, P)`, `_violations_exact(opt, Q) < _violations_exact(opt, P)`, `_Ctx(opt, P).score(Q)[0] < _Ctx(opt, P).score(P)[0]`, `_final_guards_ok(opt, P, Q, list(opt.kind), list(opt.areas))`, and exact equality of each preplaced row.

- [ ] **Step 2: Run the direct tests and verify RED**

Run:

```bash
uv run pytest tests/test_partner_group_bridge.py -q
```

Expected: collection or test failure because `bridge_grouping_violations` does not exist.

- [ ] **Step 3: Implement the minimal public wrapper**

Add this shape of implementation after `_final_guards_ok` so every dependency is already defined:

```python
def bridge_grouping_violations(opt, out, budget_s: float = 0.02):
    try:
        budget = float(budget_s)
        if not math.isfinite(budget) or budget <= 0.0:
            return out
        P0 = np.asarray(
            [[float(r[0]), float(r[1]), float(r[2]), float(r[3])]
             for r in out],
            dtype=np.float64,
        )
        if P0.shape != (int(opt.n), 4) or not np.isfinite(P0).all():
            return out
        grouping0 = _grouping_count(opt, P0)
        if grouping0 <= 0:
            return out
        ctx = _Ctx(opt, P0)
        kind = list(opt.kind)
        areas = list(opt.areas)
        score0, violations0 = ctx.score(P0)
        P, _score, _violations = _fix_grouping(
            ctx, P0.copy(), score0, violations0,
            kind, areas, time.time() + budget,
        )
        score1, violations1 = ctx.score(P)
        if _grouping_count(opt, P) >= grouping0:
            return out
        if violations1 >= violations0 or score1 >= score0 - 1e-12:
            return out
        if not _final_guards_ok(opt, P0, P, kind, areas):
            return out
        return [tuple(map(float, r)) for r in P]
    except Exception:
        return out
```

Do not call `_kill`, boundary repair, MIB repair, sliver repair, LNS, or stage-2 refinement.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_partner_group_bridge.py -q
```

Expected: all Task 1 tests pass with no new warning or stderr noise.

- [ ] **Step 5: Run the relevant existing violation/refiner tests**

Run:

```bash
uv run pytest tests/test_partner_coord_polish.py tests/test_partner_seat_frame_narrow.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add partner/violation_killer.py tests/test_partner_group_bridge.py
git commit -m "feat: add bounded grouping bridge"
```

---

### Task 2: Final-pipeline integration, warming, and diagnostics

**Files:**
- Modify: `partner/contest_optimizer.py`
- Modify: `tests/test_partner_group_bridge.py`
- Modify: `tests/test_partner_tag_compress.py`

**Interfaces:**
- Consumes: `bridge_grouping_violations(opt, out, budget_s)` from Task 1 and existing `tag_compress(opt, out)` / `warm_dependencies()`.
- Produces: independent `PARTNER_GROUP_BRIDGE`, `PARTNER_GROUP_BRIDGE_BUDGET`, and `PARTNER_GROUP_BRIDGE_DEBUG` behavior through the existing final `_tag_compress` hook.

- [ ] **Step 1: Write failing integration tests**

Extend the environment-cleanup tuples for the three new variables and add tests named `test_both_final_flags_off_are_identity_without_scorer`, `test_group_bridge_works_with_tag_compress_off`, `test_tag_then_bridge_share_one_scorer`, `test_bridge_failure_preserves_tag_result`, `test_bridge_flag_warms_dependencies_in_constructor`, and `test_group_bridge_debug_reports_self_paired_fields`.

The scorer-count test must monkeypatch `contest_optimizer._ColumnOptimizer`, enable both flags with `direct_box=None`, patch both public operations to return their input, and assert exactly one scorer construction. The failure-containment test must patch tag compression to return a distinct accepted list, patch the grouping operation to raise, and assert the tag result is returned. The debug test must require stderr fields for elapsed milliseconds, grouping before/after, total violations before/after, HPWL before/after, bbox before/after, and committed status.

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```bash
uv run pytest tests/test_partner_group_bridge.py tests/test_partner_tag_compress.py -q
```

Expected: the new flag/warming/integration tests fail because the final hook still supports tag compression only.

- [ ] **Step 3: Extend constructor-time warming**

Keep the existing warm hook name to minimize change, but gate it on either final flag:

```python
if not (os.environ.get("PARTNER_TAG_COMPRESS")
        or os.environ.get("PARTNER_GROUP_BRIDGE")):
    return
```

Continue calling `tag_compress.warm_dependencies()` so the exact counter and `violation_killer` import occur outside evaluator-timed `solve()`.

- [ ] **Step 4: Extend the existing final hook**

Within `_tag_compress`, read `tag_on` and `bridge_on`; return `out` immediately if neither is enabled. Build or reuse one scorer. Run tag compression in its own `try/except`, retaining `out` on failure. Then run `bridge_grouping_violations` in a separate `try/except`, retaining the current stage result on failure. Parse the budget with a fail-closed fallback to `0.02`; the public wrapper remains authoritative for finite/positive validation.

When `PARTNER_GROUP_BRIDGE_DEBUG=1`, capture the pre-bridge stage and emit one stderr line containing:

```text
[gbridge] n={block_count} ms={elapsed_ms:.3f} grouping={grouping0}->{grouping1} V={violations0}->{violations1} hpwl={hpwl0:.6f}->{hpwl1:.6f} bbox={bbox0:.6f}->{bbox1:.6f} committed={0_or_1}
```

Compute every before/after diagnostic on the same scorer and pre-bridge input. `committed` means the returned bridge object/layout differs from the stage input; do not infer it from elapsed time.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_partner_group_bridge.py tests/test_partner_tag_compress.py -q
```

Expected: pass.

- [ ] **Step 6: Run the full suite**

Run:

```bash
uv run pytest
```

Expected: full pass. If the only failure is the known order-sensitive anytime-ladder test, rerun that exact node in isolation and record both outputs; do not silently call the full suite passing.

- [ ] **Step 7: Update the knowledge graph and commit**

Run:

```bash
graphify update .
git add partner/contest_optimizer.py tests/test_partner_group_bridge.py tests/test_partner_tag_compress.py graphify-out
git commit -m "feat: integrate final grouping bridge"
```

Only add graph outputs changed by the code-only update; preserve unrelated dirty graph artifacts.

---

### Task 3: G0 saved-layout mechanism gate

**Files:**
- Evidence input: `/tmp/tag_package_full100.json`
- Evidence output: `.superpowers/sdd/g01-g0-results.json`
- Evidence report: `.superpowers/sdd/g01-g0-report.md`

**Interfaces:**
- Consumes: the exact 100 `positions` from the verified 0811d evaluator JSON and the Task 1 public operation.
- Produces: official per-case and weighted G0 evidence without changing production or package files.

- [ ] **Step 1: Verify the evidence input**

Require 100 test records, 100/100 feasible, and `total_score_no_runtime` within `1e-9` of `1.152956051348`. Stop if any check fails.

- [ ] **Step 2: Replay the public wrapper**

Load the 100 cases with `scratchpad/icdc/gr_lib.py`, construct one `_ColumnOptimizer` per saved layout, call `bridge_grouping_violations(opt, positions, budget_s=0.02)`, and re-evaluate each result with `gr_lib.evaluate`. Measure only the public-wrapper call using `time.perf_counter()`.

- [ ] **Step 3: Write and adjudicate G0 evidence**

Record per case: test ID, block count, elapsed time, accepted, feasibility, before/after official cost, grouping V, total V, HPWL gap, and area gap. Compute weighted totals with `scripts.iccad2026_evaluate.compute_total_score` semantics. PASS only if all Global Constraints' G0 thresholds hold and no production source contains case IDs or saved-layout data.

- [ ] **Step 4: Independent evidence review**

Have a fresh Terra xhigh reviewer inspect the report/result JSON and return PASS/FAIL against every G0 threshold. Do not start G1 on FAIL.

---

### Task 4: G1 online paired evaluator gate

**Files:**
- Evaluator entrypoint: `scripts/probes/tag_compress_warm_wrapper.py`
- Evidence outputs: `artifacts/partner_eval/gbridge_g1_pair{1,2,3}_{on,off}.json`
- Debug logs: `.superpowers/sdd/gbridge-g1-pair{1,2,3}-on.log`
- Evidence report: `.superpowers/sdd/g01-g1-report.md`

**Interfaces:**
- Consumes: reviewed Task 2 code and the existing 0.3-second tag-compress wrapper/configuration.
- Produces: three full100 pairs with orders ON/OFF, OFF/ON, ON/OFF.

- [ ] **Step 1: Run three reversed-order pairs**

Source the repository `.env` for the promoted 0811d operating point. For ON set `PARTNER_GROUP_BRIDGE=1`, `PARTNER_GROUP_BRIDGE_BUDGET=0.02`, and `PARTNER_GROUP_BRIDGE_DEBUG=1`; for OFF unset all three. Invoke `scripts/iccad2026_evaluate.py --data-path ../ --evaluate scripts/probes/tag_compress_warm_wrapper.py --output <pair-json>` from `FloorSet/iccad2026contest` and capture each ON log.

- [ ] **Step 2: Summarize and adjudicate**

Record every arm's feasibility count, no-runtime score, average runtime, ordering, and output path. Compute all three paired score/runtime deltas and their means. Parse ON diagnostics and verify every committed line has decreasing grouping and total V with no hard-guard failure.

- [ ] **Step 3: Independent evidence review**

Have a fresh Terra xhigh reviewer inspect the six JSONs, three ON logs, and report. G1 PASS requires every Global Constraints' G1 threshold. If any threshold fails, retain evidence and stop without modifying the verified package.

---

### Task 5: Conditional package promotion and final verification

**Files:**
- Modify only after G1 PASS: `submission/cadc1013/violation_killer.py`
- Modify only after G1 PASS: `submission/cadc1013/op_src.py`
- Modify only after G1 PASS: `submission/cadc1013/op_wrapper.py`
- Create: dated archive under `submission/`
- Create: `docs/experiments/2026-08-12-fast-grouping-bridge-promotion.md`

**Interfaces:**
- Consumes: reviewed G0/G1 PASS evidence.
- Produces: fresh-extracted, source-closed submission archive or no package change on gate failure.

- [ ] **Step 1: Synchronize reviewed sources and enable the flag**

Copy the reviewed partner sources byte-for-byte to their package counterparts and add `"PARTNER_GROUP_BRIDGE": "1"` to the package wrapper defaults. Do not change the bridge budget default unless G1 used a reviewed non-default value.

- [ ] **Step 2: Verify source closure and package hygiene**

Assert byte equality for `partner/contest_optimizer.py` vs `submission/cadc1013/op_src.py` and `partner/violation_killer.py` vs the packaged copy. Build the archive without caches, compiled kernels, logs, artifacts, or scratchpad files; record its checksum and entries.

- [ ] **Step 3: Evaluate a fresh extraction**

Extract into a new temporary directory, validate the interface, and run full100. PASS requires 100/100 feasible, no-runtime score no worse than the best promoted online ON result, and average runtime at most `0.300` seconds.

- [ ] **Step 4: Broad final review**

Generate a whole-branch review package and dispatch a fresh Terra xhigh reviewer. Fix and re-review all Critical/Important findings, rerun covering tests, then let Sol perform final source/evidence/package acceptance. Keep the 0811d archive as fallback even after successful promotion.
