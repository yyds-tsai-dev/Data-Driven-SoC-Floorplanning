# Final-sprint handoff — 2026-08-28 (deadline 2026-08-31 23:59)

Read this first. Everything below is on branch `final-sprint-0827` (NOT pushed — no
GitHub credentials on this box; `git bundle` works: `git bundle create x.bundle origin/main..final-sprint-0827`).
Full evidence log: `docs/experiments/2026-08-21-post-beta-p0-execution.md` §15–17 (§16 = beta root cause).

## 1. Shipping package (verified)

- **`submission/cadc1013_0828_final.tar.gz` — md5 `b25aaa7e2c457e5f9e62843cb3514bb0`, 1.197 GB, 34 entries (SHIP THIS).**
  Only change vs the 0827 package: the Flow prior is the tail-tilted fine-tune of v1
  (`checkpoints/flow_matching_ft0828_tailT24_300k_ema.pt`, 300k-step cosine anneal, EMA-only,
  same loader/architecture; §17x–17ab: official −0.003…−0.006 over 7 paired reps with 1/3 of v1's
  rep variance, v3 −0.008, v5 −0.012, v6 wash, runtime +0–2%) plus today's default-off code
  (psel helpers, `FLOW_CKPT_TAIL` router — both inert). Fallback: `cadc1013_0827_final.tar.gz`
  (md5 `163b885410c02ec021b6c3699d8cd62d`, flow v1).
- Built by `bash scripts/pack_cadc1013.sh <out_dir>` from the working tree (import closure of
  `partner/contest_optimizer.py` → `op_src.py`, `partner/shipping/op_wrapper.py`,
  `partner/shipping/requirements.txt`, `tests/synth_instances.py`, flow ckpt + v2 student ckpt).
  **Always rebuild with the script; a hand-assembled package once shipped stale modules.**
- Dry run 6 (08-28 11:50, fresh extract, NEW Python 3.13 venv from requirements.txt only —
  torch 2.6.0+cu124 / scipy 1.18.1 / numba 0.67.0 — official evaluator, load 36–43):
  `[selfcheck] cuda_available=True … flow_warm_latency=0.068s … seat_ts=kept 0.148 cpu_ratio=1.11 seat_r0=kept`,
  `loaded flow model step 300000`, no `[pool-fallback]`/`[flowtail]`, **1.1102, 100/100 feasible,
  avg 0.40 s, max 1.63 s, first case 0.055 s** (JSON: `artifacts/shadow/dryrun6_pack_off.json`).
  Dry run 5 (0827 package, load 29) was 1.1042 — the two are within the box's ±0.01 rep noise.
- Shipping env (in `partner/shipping/op_wrapper.py` and README): mid budget table +
  Flow-only (`FLOW_SLOTS=10 NREF=9`) + `DIRECT_SEAT_FIX=1` + `REFINE_RES_FRAC=0.45` +
  `WALL_REPAIR=1` + `FLOW_WARM=1` + `EARLY_EXIT=1` + `REFINE_SECURE_FALLBACK=1`,
  `OVERSAMPLE=1 KS_CAP=6`, TAG_COMPRESS/GROUP_BRIDGE=1, **polish OFF** (`PARTNER_COORD_POLISH` unset), `FLOW_ANTITHETIC=1`, `FLOW_CKPT_TAIL` unset (router off). Unchanged from 0827 except the flow checkpoint.
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

User targets: public < 1.08, v3/v5/v6 < 1.12. Judge by public + v3 + v5/v6. **alpha_1 dropped 08-28** (organizers: hidden p2b/b2b are not shifted; `run_gate5.sh` no longer runs it).
Pessimistic contest estimate (M = 1.45 CPU factor, gate thresholds scaled): public ≈ 1.14, v3 ≈ 1.17.

## 3. Method rules (do not skip)

- Only interleaved same-chain paired comparisons count; alternate arm order between reps
  (position confound bit us twice). Tools: `scripts/gate/run_gate5.sh <tag> KEY=VAL...`
  (official/v3/v5/v6/alpha_1; prints GATE5), `scripts/gate/run_shadow.sh <tag> <data_root> KEY=VAL...`,
  `scripts/gate/analyze_pairs.py A1 A2 -- B1 B2`, `scripts/gate/ev_rt.py --M 1.45 --D 0.6,0.7,0.8 <off jsons>`.
  run_shadow.sh exports the canonical base env INCLUDING `PARTNER_COORD_POLISH=1` — always pass
  `PARTNER_COORD_POLISH=` (empty) to keep polish off; `=0` does NOT disable it.
- **New (08-28, §17y/§17aa): a knob only enters the package after `full candidate env vs current package env` in ONE chain, ×4 with both arm orders.** Two knobs that each gated "significant" in their own 4-rep chains (polish headroom 0.6, ANTITHETIC=0) were +0.005 / runtime +8% when stacked and measured directly against the package env; ±0.01 effects are not resolvable on this box (bootstrap-over-cases CIs ignore rep-level load variance).
- `run_shadow.sh` now tees the full evaluator output to `artifacts/shadow/<tag>.log` (selfcheck lines visible); `scripts/gate/band_pairs.py` gives per-band paired deltas + column/direct-shipped counts.
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
**08-28 additions:** polish carve ×0.85 (all suites worse), polish HEADROOM 0.6 s (withdrawn, §17aa), `FLOW_SOLVER=heun` (+0.18–0.22, sampler 2× NFE closes the arm), budget table ×1.08 (raw −0.005 but runtime +5–8%), `FLOW_ANTITHETIC=0` (withdrawn, §17aa), fine-tune 90k snapshot (mid-band n=76–89 rung-0 failures → column ships, §17t), n-routed two-checkpoint prior `FLOW_CKPT_TAIL`/`PARTNER_FLOW_TAIL_MIN_N=95` (v3 +0.032 with the 300k tail; code kept default-off), second fine-tune arm (no gate time), `PARTNER_PSEL_DZ`/`_G` (value ≤0.00004).

## 5. Running / open (updated 08-28 12:00)

- Nothing is running and nothing is queued. Ladder/seat track closed (§17ad: ≤0.002 headroom); slow-CPU emulation confirms the 300k swap is safe (§17ac). GPU 2/3 are free; the fine-tune finished at step 300000
  (`artifacts/flow_ft_0828/flow_ft0828_ftv1_tailT24_w0-89_lr2e-5_wu1k_bs12_s300k/`, recipe in
  `scratchpad/flow_ft_0828/STATUS.md`; launcher/exporters copied to `scratchpad/flow_ft_0828/`).
- (Closed 08-28 13:00: user decided to focus on our own solver; the classmate-model thread is dropped.) For the record: a classmate's model reportedly scores 1.005–1.02 on the official 100.
  Interface cannot leak (target_positions carries only preplaced xywh / fixed wh); the validation
  container has 0/100 exact duplicates in the 1M corpus (`…/jobs/06af0e53/tmp/valcheck/`). Decisive
  check = their beta hidden raw (leaderboard best is 1.084) and whether `LiteTensorDataTest` was in
  their training data. If clean: get inference code + ckpt, run official + v3/v5/v6 on this box, then
  integrate as a candidate source in the pool (legalizer keeps feasibility/runtime).
- Push of the branch (needs a PAT/SSH key or the bundle `submission/final-sprint-0827.bundle`).

## 6. Where the remaining score is (public ≈1.10, weighted excess ≈0.10)

tail (n≥102) ≈0.07 = violations 0.03 (locked class: preplaced-tag wall lines with 5–14 blocks past
them; frame pinning trades HPWL for it) + area 0.018 (rung-0 frame 1.02·area_ref; constants 1.00/1.01 worse)
+ HPWL 0.015; mid band ≈0.025; n<76 ≈0.005. Model quality (Flow) is the remaining big lever, hence the fine-tune.

## 7. State at session close (08-28 ~13:30)

- Ship `submission/cadc1013_0828_final.tar.gz` (md5 `b25aaa7e…`); before uploading (8/30–31) re-run the
  dry-run recipe above once more on a quiet box and check the selfcheck line. Fallback = 0827 package.
- Where the last ≈0.02 to the 1.08 target would have to come from: the column-shipped class (official 27/100,
  v3 31/100, weighted excess ≈+0.009 each; §17ad) — a candidate-quality (model) problem. Every solver-side
  knob family has now been measured to the noise floor; do not re-open them.
- Evidence for every decision today: `docs/experiments/2026-08-21-post-beta-p0-execution.md` §17o–17ab;
  gate JSONs + full logs in `artifacts/shadow/{p3*,n*,ft*,rt*,fc*,fd*,po*,pl300*}_*.{json,log}`;
  chain scripts archived in `scratchpad/gate_chains_0828/`.
- Official QA 0827 (Q17–Q29) summarised in §17u: same hidden set as beta; op_wrapper used as shipped;
  init warm-up untimed; eval box driver 580 / CUDA 13.0 / torch 2.12+cu130 — our pinned torch 2.6.0
  (cu124) is compatible; pin everything in requirements.txt (done).
