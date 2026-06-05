# Quality Portfolio v1 Evaluation

## Decision

Rejected as a default path for now. Keep it as an opt-in no-runtime ablation with `FLOORSET_ENABLE_QUALITY_PORTFOLIO=auto` or `1`.

The portfolio proved there is non-GNN HPWL headroom on ID 99, but the extra post-refine runtime worsened local runtime-aware total. The current submission-safe checkpoint path should keep quality portfolio disabled by default.

## Configured Checkpoint

| Run | Total | No-runtime | Feasible | Avg runtime | Median runtime | P90 runtime | ID 99 no-runtime | ID 99 runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline high-risk normal-only | `2.3086` | `2.0362` | `100/100` | `1.04s` | `1.05s` | `1.99s` | `2.0040` | `2.15s` |
| Quality Portfolio v1, `20x48` | `2.5615` | `2.0311` | `100/100` | `0.97s` | `0.87s` | `1.84s` | `1.9959` | `2.82s` |
| Quality Portfolio v1, `12x48` | `2.4574` | `2.0332` | `100/100` | `1.48s` | `1.40s` | `2.74s` | `1.9994` | `3.85s` |

## Tail Findings

ID 99 improved because HPWL gap dropped from `0.5745` to `0.5602` in the `20x48` run while area gap and soft violations stayed unchanged. ID 98 did not benefit, so the trigger was narrowed to extreme-density / pin-heavy tail cases rather than block count alone.

The first implementation expanded full decoder candidates and was too slow: ID 99 took `81.14s`. The revised implementation refines only the best base placement and reduced ID 99 single-case runtime to about `2.96s`, but the full local runtime-aware score still regressed.

## Next Step

Use this as a search probe, not a default. The next non-GNN attempt should move the accepted ID 99 HPWL improvements into a cheaper decoder or repair primitive rather than running frontier post-refine as a portfolio candidate.
