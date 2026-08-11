# Constructive G0/G1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development while implementing each task, and superpowers:verification-before-completion before reporting results.

**Goal:** Build and measure an isolated constructive floorplanning prototype that tests whether dynamic block order plus joint shape/region decisions can close the quality gap without exceeding the final submission runtime budget.

**Architecture:** A NumPy-only module under `partner/` consumes the same raw tensors as the final partner path and emits hard-legal candidates. It differs from the retired `src/floorset_arch/constructive.py` beam by assigning MIB shapes jointly, placing cluster members consecutively against a live group frontier, exposing four materially different dynamic order policies, and using oracle information only through an explicitly quantized G0 representation. A separate offline probe scores candidates with the official evaluator; production defaults remain untouched unless both gates pass.

**Tech Stack:** Python 3.12, NumPy, PyTorch only at the dataset boundary, pytest, the existing ICCAD v10 evaluator, and `partner.icdc.data`/`partner.icdc.engine` contracts.

## Global Constraints

- Do not modify the evaluator-facing production path during G0/G1.
- Do not revive or silently route through `src/floorset_arch/constructive.py`; its prior beam result is the historical control.
- No validation label geometry may enter G1 construction or policy ranking.
- G0 may consume repaired-golden geometry only after quantizing it to order, aspect, and coarse region codes; exact rectangle passthrough is calibration-only and cannot satisfy the gate.
- Every emitted candidate must preserve fixed/preplaced dimensions, preplaced origins, soft-block area tolerance, and non-overlap before official evaluation.
- Candidate generation must be deterministic for identical inputs and policy.
- G0 pass: all 100 repaired-golden cases hard legal, weighted no-runtime score at most `1.03`, and n=120 decode time at most `20 ms`.
- G1 pass on the 21 n>=100 cases: at least four portfolio wins over repaired production, full-total-equivalent gain at least `0.005`, and at least 90% of policy candidates hard legal.
- If either gate fails, record the failure and keep the module opt-in; do not integrate it into the final wrapper.

---

## File Structure

- Create `partner/constructive_g01.py`: policy records, graph features, joint shape assignment, sparse legal placement, and gate aggregation helpers.
- Create `tests/test_partner_constructive_g01.py`: focused TDD coverage for hard legality, compound constraints, policy diversity, determinism, quantization, and gate math.
- Create `scripts/probes/constructive_g01_probe.py`: official-data G0/G1 runner and JSON evidence writer.
- Create `docs/experiments/2026-08-11-constructive-g0-g1.md`: measured gate result and decision.

### Task 1: Public Contracts and Hard-Legal Skeleton

**Files:**
- Create: `tests/test_partner_constructive_g01.py`
- Create: `partner/constructive_g01.py`

**Interfaces:**
- `ConstructivePolicy(name, order_mode, bbox_weight, hpwl_weight, anchor_weight, constraint_weight)` describes one deterministic arm.
- `ConstructiveResult(policy, rects, order, elapsed_s, hard_legal, diagnostics)` contains one candidate.
- `construct_candidate(area, constraints, target_positions, b2b, p2b, pins, policy, *, region_codes=None, aspect_codes=None)` returns a single result in `[x,y,w,h]` order.

- [ ] Write a failing test containing a preplaced obstacle, a fixed block, and ordinary soft blocks.
- [ ] Verify collection fails because `partner.constructive_g01` does not exist.
- [ ] Implement exact-area/fixed/preplaced shape decoding, sparse slots, vectorized overlap rejection, and an always-legal right-shelf fallback.
- [ ] Verify the test passes and `partner.icdc.engine.verify_hard_legal` accepts the output.
- [ ] Commit the green hard-legal skeleton.

### Task 2: Joint MIB Shapes and Cluster Frontier

**Files:**
- Modify: `tests/test_partner_constructive_g01.py`
- Modify: `partner/constructive_g01.py`

**Interfaces:**
- `assign_shapes(...) -> np.ndarray` assigns a common exact shape to compatible MIB members before placement.
- Cluster scheduling keeps a selected group active until all unplaced members are emitted.
- Each non-root cluster member must use a legal edge-abutting slot when one exists; diagnostics report any forced disconnected fallback.

- [ ] Add a failing test where two equal-area MIB members must share dimensions.
- [ ] Add a failing test where a three-member cluster must form one edge-connected component around an obstacle.
- [ ] Implement pre-placement MIB aspect synchronization and consecutive cluster-frontier expansion.
- [ ] Verify exact shape equality, connectivity, non-overlap, and hard legality.
- [ ] Commit the compound-constraint implementation.

### Task 3: Four Dynamic Order Policies and Oracle Codes

**Files:**
- Modify: `tests/test_partner_constructive_g01.py`
- Modify: `partner/constructive_g01.py`

**Interfaces:**
- `default_policies()` returns `net_closure`, `constraint_first`, `large_first`, and `pin_gravity` in stable order.
- `encode_oracle_codes(rects, *, region_bins=16)` returns order, exact-area log-aspect codes, and quantized normalized center codes; it never returns source coordinates.
- `construct_portfolio(...)` returns one deterministic candidate per requested policy.

- [ ] Add failing tests proving the four policy names are stable and at least two policies choose different first blocks on a synthetic graph.
- [ ] Add a failing test proving two oracle layouts within the same region bin encode to identical region codes.
- [ ] Implement dynamic placed-neighbor closure, constraint urgency, area, and pin-gravity ranks plus deterministic tie breaking.
- [ ] Implement quantized oracle code encoding/decoding and portfolio generation.
- [ ] Run all focused tests and commit the policy portfolio.

### Task 4: Gate Aggregation and Offline Probe

**Files:**
- Modify: `tests/test_partner_constructive_g01.py`
- Create: `scripts/probes/constructive_g01_probe.py`

**Interfaces:**
- `weighted_score(costs, block_counts)` matches the evaluator's `exp(n/12)` aggregation.
- `summarize_g1(rows, all_block_counts)` returns wins, candidate legal coverage, tail weighted scores, full-total-equivalent gain, and pass/fail.
- Probe CLI accepts `--mode {g0,g1,both}`, `--golden-layouts`, `--production-layouts`, `--out`, and optional case limits.

- [ ] Add failing synthetic aggregation tests, including full-100 denominator behavior for a 21-case tail subset.
- [ ] Implement aggregation helpers.
- [ ] Implement evaluator/case loading via `partner.icdc.data`, external JSON layout loading, per-policy timing, official scoring, and atomic JSON evidence output.
- [ ] Verify `--help` and a two-case smoke run.
- [ ] Commit the gate harness.

### Task 5: Run G0, Then G1 Only If Representation Capacity Is Credible

**Files:**
- Create: `docs/experiments/2026-08-11-constructive-g0-g1.md`

- [ ] Run G0 on all 100 repaired-golden cases and record hard legality, score, per-case timing, and n=120 timing.
- [ ] If G0 misses badly, inspect attribution once; make only a mechanism-level correction covered by a failing test, not a parameter sweep.
- [ ] Run G1 on all 21 n>=100 cases for four policies and compare best-of-four against repaired production.
- [ ] Record wins, losses, legal coverage, full-total-equivalent gain, runtime distribution, and worst contributors.
- [ ] Mark the architecture `GO`, `HOLD`, or `NO-GO`; do not modify production defaults in this task.

### Task 6: Verification and Graph Refresh

**Files:**
- Modify only if required by discovered regressions.

- [ ] Run `uv run pytest tests/test_partner_constructive_g01.py -q`.
- [ ] Run relevant existing partner/ICDC tests plus the known pre-existing failing test separately.
- [ ] Run `uv run pytest -q` only if the focused suite does not reveal a prohibitive runtime; distinguish the known baseline failure from new failures.
- [ ] Run `graphify update .` from the worktree.
- [ ] Review `git diff --check`, `git status`, and the experiment evidence before reporting.
