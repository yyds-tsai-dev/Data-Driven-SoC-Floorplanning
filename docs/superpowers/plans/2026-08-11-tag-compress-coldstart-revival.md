# Tag-compress cold-start revival

## Goal

Re-evaluate the dormant `PARTNER_TAG_COMPRESS` pass under the exact rework
condition left by its original NO-GO: keep its layout-quality mechanism, but
remove the lazy-import cost from the evaluator's timed `solve()` call.

This is not a fresh replay of the old arm. The old isolated-tail measurements
charged module initialization to the case. A saved-layout replay on the current
production outputs shows a deterministic `-0.004829` no-runtime delta, while
the hot pass itself takes about 7--8 ms on the cases it changes. An isolated
case-89 run shows a roughly 0.4 s cold post-pass penalty, so import placement is
the variable under test.

## G0

1. Replay tag compression on at least four independently generated full-100
   production layout sets.
2. Re-score every result with the official evaluator and verify 100/100 hard
   feasibility.
3. Require a mean self-paired no-runtime delta at most `-0.004`, with no
   positive replicate.

The self-pair is noise-free with respect to SA: OFF and ON use the exact same
saved input layout.

## G1

1. Add a default-off constructor warm-up that imports `tag_compress` and its
   exact violation counter before the evaluator starts timing any `solve()`.
2. Keep the pass itself and all acceptance/legality checks unchanged.
3. Add tests for flag-off identity, opt-in warm-up, failure containment, and
   unchanged compression output after warming.
4. Run paired full-100 evaluations with reversed arm order.

Promotion requires:

- 100/100 hard feasible in every arm;
- mean paired no-runtime delta at most `-0.004` and no sign reversal larger
  than 0.002;
- average runtime at most 0.300 s;
- the warm path must not execute when `PARTNER_TAG_COMPRESS` is unset.

If the average runtime remains above 0.300 s after moving initialization out
of `solve()`, retain the code default-off and record NO-GO.
