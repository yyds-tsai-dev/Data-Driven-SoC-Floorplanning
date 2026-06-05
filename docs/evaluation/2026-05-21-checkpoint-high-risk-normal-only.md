# Checkpoint High-Risk Normal-Only Ablation

## Purpose

Reduce runtime-aware total score on the configured Anchor-GNN path without sacrificing the checkpoint's placement quality. The previous high-risk portfolio spent extra repair passes on the dominant tail cases, especially ID 99, and the runtime factor dominated total score.

## Runs

| Run | Total score | No-runtime total | Feasible | Avg runtime | Median runtime | P90 runtime | Max runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Configured checkpoint baseline | `3.2697` | `2.0326` | `100/100` | `1.57s` | `1.20s` | `2.34s` | `17.63s` |
| High-risk portfolio off | `2.6763` | `2.6203` | `100/100` | `1.19s` | `1.19s` | `1.83s` | `4.39s` |
| High-risk portfolio, normal repair only | `2.3086` | `2.0362` | `100/100` | `1.04s` | `1.05s` | `1.99s` | `3.85s` |

## Tail Impact

| Test ID | Baseline cost | Normal-only cost | Baseline no-runtime | Normal-only no-runtime | Baseline runtime | Normal-only runtime |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 95 | `2.3641` | `2.3343` | `2.0652` | `2.0652` | `1.89s` | `1.59s` |
| 96 | `3.8322` | `2.5772` | `1.7906` | `1.7906` | `15.20s` | `3.55s` |
| 97 | `1.9167` | `1.7161` | `1.8596` | `1.8596` | `1.33s` | `0.81s` |
| 98 | `2.1236` | `2.0019` | `2.2158` | `2.2158` | `1.04s` | `0.75s` |
| 99 | `3.8552` | `2.4807` | `1.9983` | `2.0040` | `10.75s` | `2.15s` |

## Decision

Make high-risk repair profiles normal-only by default. This keeps the useful high-risk shape portfolio and removes the expensive repair-profile matrix from the submission path. Heavier high-risk repair profiles stay opt-in via `FLOORSET_HIGH_RISK_REPAIR_PROFILES=normal,boundary_first,grouping_first,quality_refine` for future no-runtime or server-side ablations.
