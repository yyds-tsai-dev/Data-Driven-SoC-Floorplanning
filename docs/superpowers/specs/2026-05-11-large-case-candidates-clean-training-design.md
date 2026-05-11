# Large-Case Candidates And Clean Training Design

## Goal

Improve honest `iccad2026_evaluate.py --evaluate` score toward `< 2` without changing the default checkpoint away from `checkpoints/gnn_best.pt`.

## Decisions

1. Keep `gnn_best.pt` as the default Anchor-GNN checkpoint. Current metadata and non-CLI tail-case proxy checks favor it over the 0510 larger checkpoints.
2. Add a sample-local candidate API that is structurally parallelizable, but run it sequentially by default.
3. For `block_count >= 118`, candidate generation can include only relative-order candidates: adaptive profile, forced `soft`, and forced `compact`. Full-score verification moved this matrix behind an opt-in flag because sequential default execution regressed runtime-weighted score.
4. Keep beam and no-guidance candidates opt-in only. Existing ablations showed they increased runtime without improving IDs 96-99.
5. Add repair profiles for large cases. The default profile is normal repair. Large-boundary repair is opt-in because full validation showed it reduces soft counts but loses too much runtime-weighted score.
6. Select candidates soft-count-first, then proxy-cost. A candidate with fewer boundary/group/MIB violations may win even if its area/HPWL proxy is slightly worse; equal-soft candidates use `_proxy_cost()`.
7. During supervised training, skip any sample whose `fp_sol` violates Boundary, Group, or MIB constraints. Do not train anchor/aspect/priority/order/pairwise losses on soft-violating golden answers.
8. Add a repo-root `.env` with every known `FLOORSET_*` knob and conservative defaults. Eval scripts should source it when present, while Python code continues to read normal environment variables.

## Architecture

`ArchitectureV3Optimizer.solve()` remains the production entrypoint. It will ask a small candidate builder for candidate specifications, deduplicate equivalent concrete profiles, execute them sequentially, repair each candidate under the requested repair profile, then choose the best result through a single scoring function. The large-case matrix is opt-in through `.env`; the default production path remains the faster adaptive single-candidate path.

Training clean-sample filtering belongs in `floorset_arch.training.losses` as a reusable helper and in `train.py` as a lightweight skip before forward/backward. The filter converts `fp_sol` into a `Placement` and reuses `soft_violation_counts()` so the definition stays aligned with current solver semantics.

Environment visibility is documentation plus script behavior, not a runtime dependency. `.env` is safe to source in shell scripts, and contest-facing Python remains dependency-free.

## Testing

- Unit-test candidate spec generation so large cases include adaptive/soft/compact relative-order candidates and small cases keep the existing fast path.
- Unit-test soft-count-first candidate selection independently of model loading.
- Unit-test clean training sample detection for boundary/group/MIB clean and dirty examples.
- Unit-test that eval scripts source `.env` through shell syntax.
- Run focused tests first, then the repository pytest suite.

## Self-Review

- Placeholder scan: no TBD/TODO placeholders remain.
- Internal consistency: default remains sequential and `gnn_best.pt`; opt-in beam/no-guidance stay out of default.
- Scope check: the work is one implementation slice across optimizer, training filter, scripts, and env documentation.
- Ambiguity check: soft-violating `fp_sol` means any non-zero Boundary, Group, or MIB violation count.
