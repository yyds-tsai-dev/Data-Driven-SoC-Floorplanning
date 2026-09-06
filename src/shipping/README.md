# Shipping env (2026-08-27, final-sprint-0827)

Files here are the tracked copies of what the final package needs
(`submission/` is git-ignored; package built by `scripts/pack_cadc1013.sh`):

- `budget_table_mid.txt` — `PARTNER_BUDGET_TABLE` (100 per-n seconds, n=21..120);
  fast EV table with n=76..101 raised to the model-arm unlock line + 0.07 s.
- `requirements.txt` — complete third-party set for the partner path
  (torch / numpy / numba / scipy) plus tqdm for the evaluator.  The beta
  package shipped a 0-byte file, which left scipy missing and the model
  channel dead on the contest box (see docs/experiments/2026-08-21-post-beta-p0-execution.md §16).

Env on top of the canonical partner base (run_shadow.sh / g1_runtime._SOLVER_ENV):

    PARTNER_BUDGET_TABLE=$(cat src/shipping/budget_table_mid.txt)
    DIRECT_CKPT=<v2 student: artifacts/icdc_topology_prior_training/checkpoints_s2_20k/best.pt>
    FLOW_CKPT=<flow_matching_v1_final.pt>
    PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10
    PARTNER_REFINE_RES_FRAC=0.45 PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 PARTNER_REFINE_SECURE_FALLBACK=1
    PARTNER_OVERSAMPLE=1 PARTNER_KS_CAP=6 PARTNER_TAG_COMPRESS=1 PARTNER_GROUP_BRIDGE=1
    # PARTNER_COORD_POLISH deliberately UNSET (off): raw -0.014 but a post-deadline
    # +0.25 s on every n>=95 case -> runtime-aware total worse in every M/D scenario
    # (doc §16d).  Add PARTNER_COORD_POLISH=1 only if raw score is the target.

Packaging check: the eval log must show `[selfcheck] cuda_available=True device=cuda`.

Package module list = import closure of src/solver/contest_optimizer.py (26 modules, shipped as
op_src.py + helpers) PLUS `src/solver/synth_instances.py` (op_wrapper's `_warm_jit_kernels`
builds a synthetic n=21 instance to JIT-compile the SA kernel in the untimed load window;
without it the first evaluated case pays ~10 s of numba compilation).
