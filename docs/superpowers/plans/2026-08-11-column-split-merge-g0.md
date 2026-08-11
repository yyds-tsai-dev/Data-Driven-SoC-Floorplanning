# Column Split/Merge Large-Neighborhood G0

## Goal

Test whether the production column-SA representation has useful one-step
headroom that its single-unit relocate/swap/reorder moves cannot reach inside
the 0.3 s operating point.

The candidate move is a multi-unit large-neighborhood edit:

- split one column at a contiguous unit-order cut and insert the suffix as an
  adjacent column;
- merge two adjacent columns, evaluating both vertical unit orders.

This is not a new floorplanner. It keeps the production unit construction,
preplaced-obstacle handling, exact-area shapes, boundary scoring, numba layout
kernel, and official candidate selection unchanged.

## Why this is not a replay of a closed line

- Fixed-column-count portfolio/racing was tested, but every individual SA
  chain still changes one unit at a time.
- A sequence of relocates can empty or refill a column, but crossing from one
  useful partition to another may require accepting several intermediate
  regressions. The proposed move evaluates the completed partition directly.
- Standard sequence-pair reconstruction was rejected because it cannot
  faithfully preserve preplaced pins. This move never leaves the production
  representation.

## G0 contract

1. Add a pure candidate generator with no mutation of its input.
2. Preserve every unit exactly once and respect `L`/`R` forced-column rules.
3. Reuse `_ColumnOptimizer._evaluate`, hence the existing numba layout and
   evaluator-faithful proxy.
4. Add a default-off oracle hook after the current greedy polish. The hook
   evaluates every one-step split/merge neighbor and keeps only a strict proxy
   improvement.
5. Report candidate count, elapsed time, and proxy delta under a debug flag.

## Gate

Run the production evaluator on the 21 weighted cases with `n >= 100`, with
the oracle hook off/on from the same revision and configuration.

G0 passes only if all conditions hold:

- 21/21 hard legal;
- weighted no-runtime gain extrapolated to full-100 is at least `0.005`;
- at least 3/21 cases strictly improve;
- median oracle enumeration is at most 25 ms per worker and the n=120 maximum
  is at most 60 ms per worker.

If G0 fails, record a NO-GO and do not add the move to SA. If it passes, G1
adds a low-probability stochastic split/merge family behind a separate flag,
then runs three paired full-100 evaluations. Promotion requires mean no-runtime
delta at most `-0.005`, 100/100 feasibility, and average runtime at most 0.3 s.

## TDD order

1. Candidate invariants and forced-edge filtering.
2. Synthetic landscape where a split is the unique improving neighbor.
3. Strict-improvement/no-op behavior of the oracle selector.
4. Default-off integration contract.
5. Targeted tests, full relevant partner tests, graph update, then G0 evaluator.
