# Flow fine-tune 0829 (round 2) — model-training track STATUS

Owner: deep-reasoner. Branch `final-sprint-0827`. **No commits.**
Launched 2026-08-29 16:2x UTC. **Training is RUNNING (detached, GPU 0).**

---

## 1. Recipe decision + rationale (round 2)

**Recipe: continuation of the round-1 300k annealed EMA at the evaluator-matched
tilt temperature T=12, lr 1e-5 peak, warmup 1000, cosine→0.01x over 250k steps,
objective bit-exact to v1/round-1. One variable changed: the tilt temperature.**

1. **Init = round-1 300k (`.../s300k/final.pt`), not v1.** Round-1's EMA is the
   promoted ship candidate (§17ab: official -0.003~-0.006 over 7 reps, v3 -0.008,
   v5 -0.012, runtime flat). Continuing from it makes round 2 structurally
   "round-1 + delta": if the gate reads wash/worse we ship round 1 unchanged, so
   the downside is bounded by a gate decision, not by a lost run. Re-running from
   v1 would re-pay the 300k anneal that round 1 needed just to reach parity.
   Measured `||model-ema||/||model|| = 3.6e-4` at round-1 step 300000 (the cosine
   had annealed), so seeding `model<-model, ema<-ema` is equivalent to seeding
   from the shipped EMA — no restart-shock decision to make.
2. **Tilt T=24 -> T=12 is the only lever with evidence behind it.** §17x's band
   decomposition shows round-1's entire gain sat in n=105-120 (-0.0044/-0.0042/
   -0.0102/+0.0032 across the four suites) — i.e. the tilt paid exactly where it
   was aimed. T=12 makes file sampling proportional to the official case weight
   `exp((n-120)/12)` (evaluate.py:581-603), i.e. exact importance sampling of the
   metric. Measured band shares over the 8100 training files (worker_0..89):

   | T | n<76 | 76-89 | 90-104 | 105-120 | n>=109 |
   |---|---|---|---|---|---|
   | uniform | 0.559 | 0.137 | 0.145 | 0.160 | 0.119 |
   | 24 (round 1) | 0.145 | 0.120 | 0.236 | 0.498 | 0.402 |
   | 16 | 0.061 | 0.082 | 0.219 | 0.639 | 0.532 |
   | **12 (round 2)** | **0.024** | **0.050** | **0.184** | **0.742** | **0.636** |

3. **Why the bold T and not T=16/18.** Round 1 moved the tail share by +0.34
   absolute from a cold start at 2e-5/300k and bought only -0.005 official — at
   the evaluator noise floor (0.0064, v2-postmortem Finding 5). A warm, low-LR
   continuation with a +0.14 shift (T=16) is near-certain to land inside noise and
   waste the last gate slot. T=12 is the largest shift the corpus supports
   (+0.244 absolute, 72% of round-1's move). Gating is paired and promotion needs
   sign-consistency, so an ambitious candidate has strictly better expected value
   than a timid one.
4. **Mid-band safety (the 90k failure mode, §17t).** The n=76-89 collapse was a
   *snapshot* artifact: the un-annealed 90k model failed rung-0, emptied the
   direct channel and shipped column (1.256 vs 1.116); the annealed 300k EMA
   repaired every one of those cases (§17x). Round 2 keeps all three repair
   mechanisms — full cosine to 0.01x **completed inside the budget**, EMA 0.9998,
   and a peak LR half of round-1's — and adds a smaller total parameter travel
   (lr x steps x mean-cosine = 1.25 vs round-1's 3.0). Note the sampling numbers
   also say the mid band was never the starved band: T=24 only moved 76-89 from
   0.137 to 0.120; the mass came out of n<76 (0.559->0.145). n<76 contributes
   +/-0.0006 in §17x's band table and is column-dominated anyway, so spending it
   is cheap. **No sampler floor/mixture was added**: mix(0.25, T12+T24) lands on
   (tail 0.681, mid 0.068) which is indistinguishable from plain T=14 — a new knob
   would buy nothing over choosing T, so the round-1 sampler is reused verbatim.
5. **Objective untouched** (`--x0-time-weighting none --term-t-prob 0.0`, no
   `--mib-hinge`, `--hpwl-loss-weight 0.30`): bit-exact v1. The v2 postmortem
   (+0.0268) is the record of what objective surgery does here, and there is no
   gate budget to debug a second variable.
6. **Data hygiene unchanged**: `FT_MAX_WORKER=89` keeps the whole shadow holdout
   (worker_90..99, 100,800 rows) out of training, so v3/v5/v6 stay usable and
   `official` remains the clean read.

## 2. Smoke (GPU 0, 200 steps, box load 24)

`artifacts/flow_ft_0829/smoke.log`: **13.9-14.7 it/s**, 4.6 GiB GPU, sampler
printed `P(n>=109)=0.636 P(n>=97)=0.872 P(n<=60)=0.006`, resume confirmed
(`loss 0.17-0.29 / pos_l1 0.012-0.023`; a from-scratch run is at 2.6 / 0.73).
Loss sits above round-1's step-350 reading (0.138 / 0.0096) because the batches
are now far more tail-heavy — expected, not a regression.

## 3. Live run

* Trainer PID **438010** (`setsid` detached; 8 dataloader workers 438231-438237).
* Log: `/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning/artifacts/flow_ft_0829/train.log`
* Ckpt dir: `artifacts/flow_ft_0829/flow_ft0829_ft300k_tailT12_lr1e-5_wu1k_bs12_s250k/`
* Cadence: `latest.pt` every 5k, permanent `step_XXXXXXXX.pt` every 25k,
  `keep_recent 3`, `final.pt` at 250k. (`final.pt` currently holds the **step-200
  smoke**; do not export it before the run ends — check the step number.)
* Throughput 12.9-13.5 it/s steady -> **250k ETA ~2026-08-29 21:45 UTC**
  (worst case at 8 it/s: 01:00 UTC 08-30). Deadline 08-30 10:00 UTC.

## 4. Commands

```bash
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
# launch / resume (bare re-run auto-resumes from latest.pt):
setsid nohup bash artifacts/flow_ft_0829/run_ft.sh > artifacts/flow_ft_0829/train.log 2>&1 &
# watch:
tail -f artifacts/flow_ft_0829/train.log
```
`run_ft.sh` sets `CUDA_VISIBLE_DEVICES=0`, `FT_TAIL_TEMP=12`, `FT_MAX_WORKER=89`,
reuses `artifacts/flow_ft_0828/{ft_launch.py,file_n_index.json}`, and appends any
extra CLI args (`bash run_ft.sh --max-steps 200` = the smoke).
**Do not pass a different `--lr` on resume**: the seed's optimizer carries
`base_lrs=[1e-5]` and `LambdaLR.load_state_dict` overwrites the CLI value.
`--warmup` / `--max-steps` DO take effect on resume.

## 5. Export the gate checkpoint

Gate the artifact that would actually ship = **EMA-only**, same as round 1.

```bash
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
CK=artifacts/flow_ft_0829/flow_ft0829_ft300k_tailT12_lr1e-5_wu1k_bs12_s250k
uv run python scratchpad/flow_ft_0828/export_ema_only.py \
  $CK/step_00250000.pt \
  artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt
# prints "... step=250000 method=flow_matching_v3 bytes=~430M" -- verify step.
```
Mid-run snapshot (only if the schedule slips): swap `step_00250000.pt` for
`step_00225000.pt`, but an **un-annealed** snapshot is the §17t failure mode —
prefer waiting for the anneal to finish. `scratchpad/flow_ft_0828/export_gate_ckpt.py`
also works (keeps `model` too, 860 MB) but writes a round-1 default filename, so
always pass an explicit output path.

Gate arm: `FLOW_CKPT=$PWD/artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt`
against the packed baseline `checkpoints/flow_matching_ft0828_tailT24_300k_ema.pt`,
current pack env, arms interleaved, >=2 reps each. **Do not run gates from this
track** (one evaluator at a time; GPU 3 is the scheduler's).

## 6. Reading the gate (what to look at beyond the total)

1. `official` raw paired delta is the clean read (both models exclude the shadow
   rows; v3/v5/v6 are biased *toward* the round-1 baseline only via v1, which is
   no longer an arm — round-1 and round-2 have identical hygiene, so all four
   suites are clean for this comparison).
2. Per-band deltas (<76 / 76-89 / 90-104 / 105-120) and **column-shipping counts
   per band** (distinct left-x in `positions`: column ~14-26, refined direct
   ~0.6n). A mid-band column-shipping jump = the §17t collapse returned -> reject.
3. runtime avg/max must not rise; check `[selfcheck] flow_warm_latency` ~0.07 s.
4. Promotion criterion: official does not regress **and** the shadow suites are
   sign-consistent or better; otherwise keep the 0828b pack as shipped.

## 7. Results

_(none yet)_

| tag | step | off | v3 | v5 | v6 | avg_rt | feasible | verdict |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
