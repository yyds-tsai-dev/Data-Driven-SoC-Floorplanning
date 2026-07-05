# Slack-Redistribution Refiner — Implementation Spec (2026-07-05)

Status: approved for Step-1 implementation. Origin: deep-reasoner design pass,
cross-checked by Codex peer review. Production backbone context: column-slicing
legalizer (`src/floorset_arch/legalizer/`), measured total_score_no_runtime
1.2428, 100/100 feasible (artifacts/eval_v11/phase0_column_backbone.json).

## Decision

Fixed-topology, fixed-dimension coordinate refiner in `src/floorset_arch/refine/`,
gated by `FLOORSET_SLACK_REFINE` (default off). Phase 1 keeps every block's (w,h)
exactly as the backbone produced and only translates blocks: two decoupled 1-D
convex problems (all x with y fixed, then swap), solved by DAG separation
constraints + per-block weighted-median targets + projected Gauss-Seidel sweeps.
Pure numpy, single-core, deterministic, feasibility-preserving by construction
(never changes relative order on either axis, never changes dims). Worst case
no-op: refiner is a pure function that returns its input on any doubt.

Phase 2 (aspect moves, `FLOORSET_SLACK_REFINE_ASPECT`) and Phase 3 (bbox shrink)
are separately gated follow-ons. Phase 2 is where the column-representation
ceiling (~1.35x GT HPWL) actually breaks.

## 1. Constraint-graph extraction (from legal layout)

Inputs: `rects` from backbone + raw tensors (area_targets, constraints,
target_positions, b2b, p2b, pins).

Block classification: `kind[i] in {SOFT, RIGID, LOCKED}` (LOCKED = preplaced
exact x/y/w/h; RIGID = fixed-shape dims; SOFT = area-only). `dims[i]` frozen in
Phase 1.

Pair-to-axis assignment: for each pair compute current-geometry projection
overlaps `ox`, `oy`. If y-intervals overlap by > SEP_TOL → x-edge in Gx from
smaller-x block L to R with gap w_L (x_R >= x_L + w_L). Else if x-intervals
overlap → y-edge in Gy bottom→top with gap h_bottom. Corner-only diagonal
neighbours: NO edge (transitively separated; adding both axes over-constrains).
Ties (equal coordinate) break by index to guarantee acyclicity. Keep all O(n^2)
edges (n<=120); transitive reduction only if profiling demands.

Never flip an order relation (topology-preservation invariant). Abutment/
grouping preservation is the guard layer's job (§4), not the graph's.

LOCKED blocks: pinned nodes (lo == hi == exact target values copied from
target_positions, never recomputed).

Boundary-tagged blocks and the bbox: Phase 1 FREEZES bbox extents
(X0,X1,Y0,Y1 = backbone values) as pinned virtual wall nodes WL/WR/WB/WT.
left-tag → x_i == X0; right-tag → x_i + w_i == X1; top/bottom symmetric.
Boundary blocks therefore never move off their wall; boundary guarantees and
Area_gap are preserved by construction. Bbox shrink deferred to Phase 3.

## 2. Optimization (per axis)

Objective: min sum W_ij * |cx_i - cx_j| (b2b) + sum W_pi * |px_p - cx_i| (p2b),
cx_i = x_i + w_i/2 (evaluator's exact centroid HPWL). Subject to
x_v >= x_u + g_uv for every Gx edge; pinned nodes fixed.

Projected coordinate descent, Gauss-Seidel:

```
for sweep in range(MAX_SWEEPS ~ 12, early-exit when no block moved):
    for i in topo_order(Gx) (alternate direction each sweep):
        if pinned[i]: continue
        t = weighted_median(anchor centroids of i's b2b/p2b partners, weights)
        lo_i = max(pin_lo, max over preds u of x_u + g_ui)
        hi_i = min(pin_hi, min over succs v of x_v - g_iv)
        x_i = clamp(t - w_i/2, lo_i, hi_i)
```

Weighted median (not mean) is the exact 1-D minimizer; O(deg log deg) per block.
Separable convex + difference constraints → sweeps converge to global optimum
for the fixed topology; 3-8 sweeps suffice. Axis assignment is fixed once at
extraction (do NOT re-derive DAGs after moves). x and y therefore fully
independent → one x-solve + one y-solve is optimal; MAX_ROUNDS=1.

No bbox-area term in Phase 1 (bbox pinned). Budget: <50ms single-core per case.

## 3. Phase 2 — aspect moves (separate gate FLOORSET_SLACK_REFINE_ASPECT)

SOFT blocks only: propose w' = clamp(w*sqrt(rho), slack bounds) for
rho in {0.8, 0.9, 1.11, 1.25}, h' = area/w' (machine-exact), stay < 0.9% area
error (evaluator allows 1%). Joint (w,x)/(h,y) update projected on DAG bounds;
re-clamp all bounds after any accepted dim change. MIB groups: one shared (w,h)
literal for the whole group (evaluator compares round(w,4),round(h,4) sets).
RIGID/LOCKED never aspect-moved. Accept only if guarded HPWL strictly improves.

## 4. Soft-constraint guards (evaluator-faithful, run before any accept)

Baseline triple (V_boundary, V_grouping, V_mib) computed on incoming layout;
candidate must be componentwise <= baseline.

- Boundary: recompute bbox; tag checks with the evaluator's exact 1e-6 tol.
- Grouping: shapely unary_union of member boxes per group; count
  len(geoms)-1. Use shapely directly (project dep, bit-identical to evaluator;
  do NOT reimplement with epsilon rules). Phase-1 mitigation: treat all blocks
  of a movable cluster group as a rigid sub-assembly (translate together);
  re-check with shapely on accept.
- MIB: distinct (round(w,4), round(h,4)) per group - 1. Phase 1 trivially 0 delta.

guards.py: pure `soft_violations(positions, constraints) -> (int,int,int)`.

## 5. Acceptance (no GT baselines inside solve())

Baselines are positive per-case constants → raw improvements imply gap
improvements. Accept iff:
HPWL_new < HPWL_old - EPS, bbox_new <= bbox_old (+0 in Phase 1), all three soft
counts <= baseline, hard_legal(new).

## 6. Failure containment

`refine_layout(rects, ...) -> rects` pure function; whole body in try/except;
final hard-legality re-check mirroring evaluator (overlap >1e-6 both axes; soft
area |wh-a|/a <= 0.01; fixed/preplaced dims/pos within 1e-4). Any exception,
failed re-check, or deadline hit → return ORIGINAL rects. Backbone's 1.2428 is
the floor.

## 7. Numerics policy

| Check | Evaluator tol | Refiner policy |
|---|---|---|
| Overlap | >1e-6 both axes | projection enforces exact separation; touching legal |
| Soft area | <=0.01 rel | Phase 1 dims frozen; Phase 2 cap 0.9%, h = a/w exact |
| Fixed/preplaced dims | 1e-4 | never touched; LOCKED pinned to target literals |
| Preplaced pos | 1e-4 | pinned, outside movable set |
| MIB shape | round(w,4) equality | shared literal across group |
| Boundary touch | 1e-6 | wall-pinned; copy wall literal exactly |

Emit raw float64 positions; never round emitted coordinates.

## 8. Expected gain

Phase 1: recover 20-35% of hpwl_gap on n>=100 (0.16-0.38 → ~0.11-0.27) ≈
total ~1.21-1.23. Estimate only — decided by per-case (HPWL_before, HPWL_after)
log + eval gate. If Phase-1 gain <1%, its value is as the safe foundation for
Steps 2-3 (aspect is the real ceiling-breaker toward hpwl_gap ~0.08).

## Module layout

```
src/floorset_arch/refine/__init__.py   # exports refine_layout
src/floorset_arch/refine/api.py        # orchestration + failure containment
src/floorset_arch/refine/constraint_graph.py  # build_axis_dags -> Gx, Gy
src/floorset_arch/refine/slack_solve.py       # project_axis sweeps
src/floorset_arch/refine/wirelength.py        # weighted_median_target, hpwl
src/floorset_arch/refine/guards.py            # soft_violations, hard_legal (shapely)
src/floorset_arch/refine/aspect.py            # Phase 2, lazy import
```

Integration (only production-path edit), in
`src/floorset_arch/legalizer/column_backbone.py::solve_with_column_backbone`:

```python
out = legalize_rectangles(seed_rects, ...)
if os.environ.get("FLOORSET_SLACK_REFINE", "0") == "1":
    from floorset_arch.refine.api import refine_layout
    out = refine_layout(out, area_targets, constraints, target_positions,
                        b2b, p2b, pins, deadline=deadline,
                        enable_aspect=os.environ.get("FLOORSET_SLACK_REFINE_ASPECT", "0") == "1")
return out
```

## Tests (tests/test_refine_invariants.py)

1. phase1 never changes dims (byte-identical w,h)
2. no overlap on refined output (all 100 val layouts or a sampled subset)
3. LOCKED x/y/w/h byte-identical to target_positions
4. V_boundary no-regress (should be ==)
5. V_grouping no-regress (shapely)
6. V_mib no-regress
7. HPWL monotone: accepted iff strictly decreased; synthetic slack case must drop
8. forced exception / infeasible seed → output == input exactly
9. Gx, Gy acyclic on val layouts
10. post-projection: all edge constraints hold to >= -1e-9
11. soft-area check still passes

Eval gate: `FLOORSET_SLACK_REFINE=1 bash scripts/eval_total.sh --output ...` →
total_score_no_runtime strictly < 1.2428, 100/100 feasible, plus per-case
raw-HPWL reduction log.

## Landing sequence

1. Step 1: Phase-1 refiner (dims/bbox frozen, groups rigid) + tests + eval gate.
2. Step 2: relax groups-rigid to per-block moves w/ shapely re-union per accept.
3. Step 3: Phase-2 aspect moves behind FLOORSET_SLACK_REFINE_ASPECT.

Authoritative evaluator references: FloorSet/iccad2026contest/iccad2026_evaluate.py
(HPWL 153-194, overlap 210-225, area 228-256, dims 259-303, boundary 519-541,
grouping 501-506, MIB 511-517); FloorSet/utils.py 149-239.
