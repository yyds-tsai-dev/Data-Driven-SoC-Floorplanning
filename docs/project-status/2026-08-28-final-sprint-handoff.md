# Final-sprint handoff — 2026-08-28 (deadline 2026-08-31 23:59)

Read this first. Everything below is on branch `final-sprint-0827` (NOT pushed — no
GitHub credentials on this box; `git bundle` works: `git bundle create x.bundle origin/main..final-sprint-0827`).
Full evidence log: `docs/experiments/2026-08-21-post-beta-p0-execution.md` §15–17 (§16 = beta root cause).

## 1. Shipping package (verified)

- `submission/cadc1013_0827_final.tar.gz` — md5 `163b885410c02ec021b6c3699d8cd62d`, 1.20 GB, 34 entries.
- Built by `bash scripts/pack_cadc1013.sh <out_dir>` from the working tree (import closure of
  `partner/contest_optimizer.py` → `op_src.py`, `partner/shipping/op_wrapper.py`,
  `partner/shipping/requirements.txt`, `tests/synth_instances.py`, flow ckpt + v2 student ckpt).
  **Always rebuild with the script; a hand-assembled package once shipped stale modules.**
- Dry run (fresh extract, clean Python 3.13 venv from requirements.txt only, official evaluator):
  `[selfcheck] cuda_available=True ... seat_ts=kept 0.148 ... seat_r0=kept`, noRT 1.1042 (load 29),
  100/100 feasible, avg 0.37 s, max 1.39 s, first case 0.051 s.
- Shipping env (in `partner/shipping/op_wrapper.py` and README): mid budget table +
  Flow-only (`FLOW_SLOTS=10 NREF=9`) + `DIRECT_SEAT_FIX=1` + `REFINE_RES_FRAC=0.45` +
  `WALL_REPAIR=1` + `FLOW_WARM=1` + `EARLY_EXIT=1` + `REFINE_SECURE_FALLBACK=1`,
  `OVERSAMPLE=1 KS_CAP=6`, TAG_COMPRESS/GROUP_BRIDGE=1, **polish OFF** (`PARTNER_COORD_POLISH` unset).
- Before uploading (8/30–31): `bash scripts/pack_cadc1013.sh /tmp/pk && cd /tmp/pk && tar xzf cadc1013.tar.gz`,
  `python3.13 -m venv .venv_eval && .venv_eval/bin/pip install -r cadc1013/requirements.txt`, run the
  official evaluator with `--evaluate cadc1013/op_wrapper.py`, and check stderr for
  `[selfcheck] cuda_available=True` and NO `[pool-fallback]` line. Upload only `cadc1013.tar.gz`.

## 2. Scores (shipping env; box is shared, absolute values drift ±0.02 with load)

| suite | quiet-box value | note |
|---|---|---|
| public (official100) | 1.09–1.10 | 1.135–1.152 at session start (08-26) |
| shadow v3 (stress) | 1.145–1.155 | |
| shadow v5 (quantile 0.20, zero-MIB) | 1.11–1.13 | best hidden proxy (see README in shadow_hidden/) |
| shadow v6 (quantile 0.50) | 1.12–1.14 | |
| alpha_1 | 1.20–1.22 | synthetic pin shift; secondary only |

User targets: public < 1.08, v3/v5/v6 < 1.12. Judge by public + v3 + v5/v6; alpha_1 only secondary.
Pessimistic contest estimate (M = 1.45 CPU factor, gate thresholds scaled): public ≈ 1.14, v3 ≈ 1.17.

## 3. Method rules (do not skip)

- Only interleaved same-chain paired comparisons count; alternate arm order between reps
  (position confound bit us twice). Tools: `scripts/gate/run_gate5.sh <tag> KEY=VAL...`
  (official/v3/v5/v6/alpha_1; prints GATE5), `scripts/gate/run_shadow.sh <tag> <data_root> KEY=VAL...`,
  `scripts/gate/analyze_pairs.py A1 A2 -- B1 B2`, `scripts/gate/ev_rt.py --M 1.45 --D 0.6,0.7,0.8 <off jsons>`.
  run_shadow.sh exports the canonical base env INCLUDING `PARTNER_COORD_POLISH=1` — always pass
  `PARTNER_COORD_POLISH=` (empty) to keep polish off; `=0` does NOT disable it.
- Always pass `PARTNER_SEAT_R0_ADAPT=0` (it is the default now; the CPU benchmark is unusable as a gate).
- Single-case runs are useless for the model arms (first-case sampler cold start closes the arm).
- One evaluator at a time on the box; chains run as `setsid nohup bash chain.sh`; GPU 3 for evals.
- `pkill -f "[c]hainX"` style patterns must not appear elsewhere in the same command line (self-kill).

## 4. Measured and rejected (keep default off)

Drop TAG+BRIDGE, COL_BALANCE, v4 student, partner tuning, mid2/mid3 tables, QUOTA_FIRST, FLOW_STEPS=4,
FRAME_SCALE 1.00/1.01, NREF=12, RES_FRAC 0.35/0.6, SEAT_R0=0.16, VKILL, ANYTIME_LADDER, TIGHTEN_FINE,
short-slice reserve, low-budget oversample, PIN_FRAME global, PIN_FRAME_SLOTS (all variants; only alpha_1 gains),
COND_P2B=off (public +0.058 — pins matter on the real distribution), polish 300/100/40 ms
(raw −0.014 but +0.07 s/case → runtime-aware worse), tail budget ×1.3–2.0 (raw −0.010, total +0.04/step),
LADDER_REBUDGET (+SECURE_MIN): v6 −0.009 but public wash with 3× rep variance.

## 5. Running / open at handoff time

- **Flow fine-tune (deep-reasoner, started 08-28 ~00:30)**: trains on GPU 2 from
  `submission/cadc1013/checkpoints/flow_matching_v1_final.pt`; outputs under `artifacts/flow_ft_0828/`
  (checkpoints named by config, EMA), plan/status in `scratchpad/flow_ft_0828/STATUS.md` (if written).
  The agent dies with the session; the training process (setsid) survives. To gate a checkpoint:
  five-suite chain, v1 vs candidate as `FLOW_CKPT=...`, alternate order, ×2; check
  `[selfcheck] flow_warm_latency` stays ≈0.07 s. Promote only if public+v3+v5/v6 improve, runtime flat.
- **chain P3** (polish carve ×0.85 table / headroom 0.6 s vs base, five suites ×2) then **chain N**
  (`FLOW_SOLVER=heun` / table ×1.08 / `FLOW_ANTITHETIC=0`, ×2): logs in
  `/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/chain{P3,N}.log`; results persist in
  `artifacts/shadow/{p3*,n*}_{off,v3,v5,v6,a1}.json` — analyse with analyze_pairs.py even if the logs are gone.
- Push of the branch (needs a PAT/SSH key or the bundle).

## 6. Where the remaining score is (public ≈1.10, weighted excess ≈0.10)

tail (n≥102) ≈0.07 = violations 0.03 (locked class: preplaced-tag wall lines with 5–14 blocks past
them; frame pinning trades HPWL for it) + area 0.018 (rung-0 frame 1.02·area_ref; constants 1.00/1.01 worse)
+ HPWL 0.015; mid band ≈0.025; n<76 ≈0.005. Model quality (Flow) is the remaining big lever, hence the fine-tune.
