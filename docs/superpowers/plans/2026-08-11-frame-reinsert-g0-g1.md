# Preplaced-Frame Reinsert G0/G1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover residual boundary score by moving only movable rectangles that overshoot an attainable wall line fixed by a boundary-tagged preplaced rectangle.

**Architecture:** A pure `frame_reinsert` module derives reusable wall targets from the instance constraints, removes the small set of movable outliers beyond one target wall, and greedily reinserts them into legal obstacle-edge slots inside the target frame. It accepts a result only when exact soft violations strictly decrease while raw HPWL and bbox area do not increase, which guarantees lower evaluator no-runtime cost without knowing golden baselines. G0 operates only on saved layouts; G1 adds a default-off final-layout hook only if the measured gate passes.

**Tech Stack:** Python 3.12, NumPy, PyTorch at the dataset seam, pytest, existing ICCAD v10 evaluator helpers.

## Global Constraints

- Work only in the isolated `feat/frame-reinsert-g0` worktree until the gate passes.
- Never use validation `test_id` inside the method; triggers come only from boundary/preplaced geometry.
- Preplaced/fixed geometry and every rectangle dimension remain bit-identical.
- Skip an outlier set containing a preplaced rectangle or a movable cluster member in G0.
- Reject every candidate with overlap, out-of-frame placement, non-finite coordinates, HPWL regression, bbox-area regression, or non-decreasing exact soft violations.
- G0 promotion gate: 100/100 hard feasible, full-score-equivalent gain at least `0.005`, at least three strict n>=100 wins, median internal runtime at most `10 ms`, and n=120 runtime at most `30 ms`.
- G1 promotion gate: two reversed full-100 pairs, mean no-runtime delta at most `-0.005`, 100/100 feasible, and average runtime at most `0.3 s`.

---

### Task 1: Target-Line and Reinsertion Core

**Files:**
- Create: `tests/test_partner_frame_reinsert.py`
- Create: `partner/frame_reinsert.py`

**Interfaces:**
- Produces: `FrameTarget(axis: int, side: int, line: float, owner: int)`.
- Produces: `frame_targets(rects, locked, boundary) -> list[FrameTarget]`.
- Produces: `reinsert_to_target(rects, target, locked, cluster, hpwl_fn) -> np.ndarray | None`.

- [ ] **Step 1: Write failing tests** for an attainable right wall, a top-wall outlier that has a legal interior slot, a locked outlier abort, and exact dimension preservation.

```python
def test_reinserts_movable_top_outlier_inside_preplaced_wall():
    rects = np.array([[0, 0, 2, 2], [0, 4, 2, 2], [2, 0, 2, 2]], float)
    target = FrameTarget(axis=1, side=1, line=4.0, owner=2)
    got = reinsert_to_target(rects, target, np.array([0, 0, 1], bool),
                             np.zeros(3, int), lambda p: float(p[:, 0].sum()))
    assert got is not None
    assert np.max(got[:, 1] + got[:, 3]) <= 4.0 + 1e-9
    np.testing.assert_array_equal(got[:, 2:], rects[:, 2:])
```

- [ ] **Step 2: Run RED** with `uv run pytest tests/test_partner_frame_reinsert.py -q`; expect import failure for `frame_reinsert`.
- [ ] **Step 3: Implement minimal target derivation and slot search.** Candidate low coordinates are the frame walls, HPWL weighted-median target, and every retained rectangle edge (`lo-size`, `hi`). Enumerate deterministic Cartesian products, reject overlaps vectorially, and choose minimum `(hpwl, displacement, x, y)`.
- [ ] **Step 4: Run GREEN** with the same pytest command; expect all Task-1 tests to pass.
- [ ] **Step 5: Commit** `partner/frame_reinsert.py` and `tests/test_partner_frame_reinsert.py`.

### Task 2: Monotone Guarded Pass

**Files:**
- Modify: `tests/test_partner_frame_reinsert.py`
- Modify: `partner/frame_reinsert.py`

**Interfaces:**
- Produces: `FrameReinsertResult(rects, attempted, accepted, elapsed_s, reason)`.
- Produces: `frame_reinsert(opt, out, *, viol_fn=None, hpwl_fn=None) -> FrameReinsertResult`.

- [ ] **Step 1: Write failing tests** proving rejection on equal/worse V, HPWL regression, bbox regression, overlap, and cluster outliers; prove acceptance on strict V decrease with non-regressing quality.

```python
def test_guard_accepts_only_strict_v_drop_without_quality_regression():
    result = frame_reinsert(fake_opt, source, viol_fn=violation_counter,
                            hpwl_fn=lambda p: 0.0)
    assert result.accepted == 1
    assert violation_counter(fake_opt, np.asarray(result.rects)) < violation_counter(fake_opt, source)
```

- [ ] **Step 2: Run RED** for the new guarded-pass tests; expect missing `frame_reinsert` interface.
- [ ] **Step 3: Implement the minimal pure pass.** Try target lines in deterministic order, retain the best lexicographic `(V, area, HPWL)` result, and return the original object on no-op or any exception.
- [ ] **Step 4: Run GREEN** and then `uv run pytest tests/test_partner_tag_compress.py tests/test_partner_frame_reinsert.py -q`.
- [ ] **Step 5: Commit** the guarded pass.

### Task 3: Saved-Layout G0 Probe

**Files:**
- Create: `scripts/probes/frame_reinsert_g0.py`
- Modify: `tests/test_partner_frame_reinsert.py`

**Interfaces:**
- CLI consumes `--layouts`, `--out`, optional `--min-n`, and the repository FloorSet path.
- JSON produces per-case before/after official costs, feasibility, metric deltas, attempted/accepted counts, and internal elapsed time plus weighted full-100 aggregation.

- [ ] **Step 1: Write a failing aggregation test** that uses the evaluator's full-100 `exp(n/12)` denominator even when only n>=100 candidates change.
- [ ] **Step 2: Run RED** and confirm the aggregation helper is missing.
- [ ] **Step 3: Implement the probe** using `partner.icdc.data.load_test_cases`, `partner.icdc.dump_bank.official_cost`, and a `_ColumnOptimizer` scorer built from each saved layout. Write output atomically through a temporary sibling followed by `os.replace`.
- [ ] **Step 4: Run GREEN**, `--help`, then a two-case smoke.
- [ ] **Step 5: Commit** the probe and aggregation test.

### Task 4: Full G0 Gate and Decision

**Files:**
- Create: `docs/experiments/2026-08-11-frame-reinsert-g0.md`

**Interfaces:**
- Consumes the final package evidence `/tmp/tag_package_full100.json`.
- Produces a GO/NO-GO decision against the Global Constraints.

- [ ] **Step 1: Run full-100 G0** and save evidence outside the repository.
- [ ] **Step 2: Verify official hard feasibility and recompute weighted score independently from per-case rows.**
- [ ] **Step 3: Attribute failures once.** One mechanism-level correction is allowed only if a new failing test captures it; no parameter sweep.
- [ ] **Step 4: Record score, wins, metric deltas, runtime distribution, and decision in the experiment document.**
- [ ] **Step 5: Commit** the experiment record.

### Task 5: Conditional G1 Production Hook

**Files:**
- Modify only after G0 GO: `partner/contest_optimizer.py`
- Modify only after G0 GO: `tests/test_partner_frame_reinsert.py`
- Modify only after G0 GO: `.env`

**Interfaces:**
- `PARTNER_FRAME_REINSERT=1` enables the final-layout pass; default off otherwise.
- Reuses the scorer and warmed exact-violation dependency already used by tag compression.

- [ ] **Step 1: If G0 is NO-GO, stop this task and leave production untouched.**
- [ ] **Step 2: If G0 is GO, write a failing hook test** proving flag-off object identity and flag-on invocation after tag compression.
- [ ] **Step 3: Run RED**, implement the minimal hook, then run GREEN.
- [ ] **Step 4: Run two reversed full-100 pairs and apply the G1 gate.**
- [ ] **Step 5: Promote and commit only if every G1 criterion passes; otherwise document NO-GO and revert no production defaults.**

### Task 6: Verification and Graph Refresh

**Files:**
- Modify only files required by a discovered regression.

- [ ] **Step 1: Run** `uv run pytest tests/test_partner_frame_reinsert.py tests/test_partner_tag_compress.py tests/test_partner_seat_final.py tests/test_partner_coord_polish.py -q`.
- [ ] **Step 2: Run** `git diff --check` and inspect `git status --short`.
- [ ] **Step 3: Run** `graphify update .`.
- [ ] **Step 4: Re-run the full focused test command after graph refresh if source files changed during verification.**

