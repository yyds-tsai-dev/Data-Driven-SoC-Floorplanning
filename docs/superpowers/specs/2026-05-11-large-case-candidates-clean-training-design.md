# Large-Case Candidates And Clean Training Design

## Goal

Improve no-runtime local quality score toward `< 2` while promoting checkpoints only by evaluator evidence.

## Decisions

1. Use full-validation `total_score_no_runtime` as the checkpoint promotion metric. The current promoted default is `gnn_latest_0510_ns200000_ep10_h192_l6_acc32.pt`.
2. Add a sample-local candidate API that is structurally parallelizable, but run it sequentially by default.
3. For `block_count >= 118`, candidate generation can include only relative-order candidates: adaptive profile, forced `soft`, and forced `compact`. Earlier full-score verification moved this matrix behind an opt-in flag because sequential default execution regressed local runtime-aware score; future promotion needs no-runtime score and raw-runtime review.
4. Keep beam and no-guidance candidates opt-in only. Existing ablations showed they increased runtime without improving IDs 96-99.
5. Add repair profiles for large cases. The default profile is normal repair. Large-boundary repair is opt-in because earlier full validation showed it reduces soft counts but loses too much local runtime-aware score.
6. Select candidates soft-count-first, then proxy-cost. A candidate with fewer boundary/group/MIB violations may win even if its area/HPWL proxy is slightly worse; equal-soft candidates use `_proxy_cost()`.
7. During supervised training, do not fully trust any sample whose `fp_sol` violates Boundary, Group, or MIB constraints. A strict clean-only policy remains available, but the default is weighted dirty-sample training because a measured 500-sample probe found only 1 clean sample. Dirty samples are low-weight geometry references and do not contribute order/pairwise supervision by default.
8. Add a repo-root `.env` with every known `FLOORSET_*` knob and conservative defaults. Eval scripts can source it, and Python loads it through `python-dotenv` without overriding explicit environment variables.

## Architecture

`ArchitectureV4Optimizer.solve()` remains the production entrypoint. It will ask a small candidate builder for candidate specifications, deduplicate equivalent concrete profiles, execute them sequentially, repair each candidate under the requested repair profile, then choose the best result through a single scoring function. The large-case matrix is opt-in through `.env`; the default production path remains the faster adaptive single-candidate path.

Training clean-sample detection belongs in `floorset_arch.training.losses` as a reusable helper and in `train.py` as policy-controlled weighting before forward/backward. The detector converts `fp_sol` into a `Placement` and reuses `soft_violation_counts()` so the definition stays aligned with current solver semantics.

Environment visibility is documentation plus script behavior plus `python-dotenv` loading. `.env` is safe to source in shell scripts, and Python reads it only as default environment values.

## Testing

- Unit-test candidate spec generation so large cases include adaptive/soft/compact relative-order candidates and small cases keep the existing fast path.
- Unit-test soft-count-first candidate selection independently of model loading.
- Unit-test clean training sample detection for boundary/group/MIB clean and dirty examples.
- Unit-test that eval scripts source `.env` through shell syntax.
- Run focused tests first, then the repository pytest suite.

## Self-Review

- Placeholder scan: no TBD/TODO placeholders remain.
- Internal consistency: default remains sequential and promoted by no-runtime evaluator evidence; opt-in beam/no-guidance stay out of default.
- Scope check: the work is one implementation slice across optimizer, training filter, scripts, and env documentation.
- Ambiguity check: soft-violating `fp_sol` means any non-zero Boundary, Group, or MIB violation count.
