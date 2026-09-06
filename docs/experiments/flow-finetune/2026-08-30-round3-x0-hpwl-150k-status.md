# Flow fine-tune 0830 (round 3) — model-training track STATUS

Owner: deep-reasoner. Branch `final-sprint-0827`. **No commits.**
Launched 2026-08-30 09:50 UTC. **RUNNING detached, GPU 0.**

## 1. Recipe + rationale

**Continuation of the round-2 250k (T=12) checkpoint, lr 5e-6 peak, warmup 1000,
cosine→0.01x over 150k steps, EMA 0.9998, tilt T=12 unchanged, and ONE objective
change: `--x0-loss-weight 1.0→2.0`, `--hpwl-loss-weight 0.30→0.60`.**

1. **Why an objective tilt at all.** The oracle probe (§18l) says the remaining
   gap is *prediction accuracy of golden geometry on n>=105*: the refine ladder
   reaches 1.013 from a golden seed but 1.072 from our Flow seed on the same 19
   heavy cases (seed hpwl_gap 0.019 / area_gap 0.034 vs 0.0002 / 0.003). Round 2
   already spent the sampling lever (T=24→12, tail share 0.498→0.742) and bought
   only official −0.001 — at the 0.0064 noise floor. A third pure continuation at
   half LR is near-certain to land inside noise and waste the last gate slot.
   The only lever left that targets the measured deficit is the loss itself.
2. **Why these two terms.** `x0_loss` is the direct coordinate endpoint error —
   literally the quantity the probe says is off (pos_l1 ≈ 0.012–0.027 vs a golden
   seed's ~0.0002 area/0.003 hpwl gaps). `hp_loss` is a **relu hinge on HPWL
   excess relative to the golden layout** (`relu(hp_pred-hp_gt)/hp_gt`,
   flow_matching_train.py:197-199) — evaluator-aligned by construction and
   one-sided, so it cannot push the model to be "better than golden" in a way
   that trades away coordinate accuracy. Doubling both re-aims gradient at
   endpoint accuracy relative to the velocity-matching term (v stays 0.5).
3. **Why half the LR.** Doubling the x0 term roughly doubles that component's
   gradient magnitude; lr 1e-5→5e-6 keeps effective step size on the tilted terms
   near round-2's, so this stays a *refinement*, not a re-training. Total travel
   (lr x steps x mean-cosine) = 0.375 vs round-2's 1.25 and round-1's 3.0.
4. **Bounded downside.** Init is the round-2 final, i.e. round 3 = round 2 + delta.
   If the gate reads wash/worse we ship the existing pack unchanged; the risk is a
   gate decision, not a lost artifact.
5. **Honest risk.** This is objective surgery, the same *class* of change that
   cost +0.0268 in the v2 postmortem. Difference: v2 changed the *t-distribution*
   (`x0-time-weighting snr`, `term-t-prob`), i.e. which noise levels get learned;
   round 3 changes only two scalar term weights and leaves t uniform
   (`--x0-time-weighting none --term-t-prob 0.0`, no `--mib-hinge`). Mitigations
   kept: full cosine anneal COMPLETES inside the budget (the §17t mid-band
   collapse was an un-annealed-snapshot artifact), EMA 0.9998, halved peak LR.
   **Mid-band (n=76-89) column-shipping counts must be checked at the gate.**
6. **Not stacked**: no sampler change, no mib-hinge, no min-snr change, no
   overlap/boundary reweight. One variable per round is the whole discipline.
7. **Data hygiene unchanged**: `FT_MAX_WORKER=89` keeps worker_90..99 (the whole
   v1/v3/v5/v6 shadow pool, 100,800 rows) out of training. `official` is clean.

## 2. Resumability proof (the round-2 `--lr` trap)

`direct_diffusion_train.main` restores `optimizer`/`sched` state on resume, so a
CLI `--lr` is overwritten by the seed's `base_lrs` — but **loss weights are read
from `args` inside `train_step` every step**, so `--x0-loss-weight` /
`--hpwl-loss-weight` DO take effect on resume. Both were still proven live:

* Seed rebuilt at step 0 via `make_seed_checkpoint.py` with `FT_SEED_SRC=` the
  round-2 `step_00250000.pt` (verified `step==250000`, method flow_matching_v3;
  round-2 `final.pt` is also a genuine 250000 — the smoke was overwritten).
  Printed `base_lrs [5e-06, 5e-06]`; `||model-ema||/||model|| = 2.6e-4`, so
  seeding model<-model is equivalent to seeding from the shipped EMA.
* Smoke lr at step 200 printed **1.00e-06** = 200/1000 x 5e-6 (round-2 weights
  would have shown 2.00e-06). Peak 5.00e-06 confirmed live after warmup.
* Loss-weight falsification on the smoke `train_log.jsonl`: the implied
  `large_case_weight = loss / sum(w_i * term_i)` is clamped to [0.5, 3] in code.
  Under the OLD weights (x0=1.0, hp=0.3) 4/5 logged rows imply 3.30–3.96 —
  impossible. Under the NEW weights (x0=2.0, hp=0.6) all 5 imply 2.23–2.72,
  consistent with n≈105-120 (`exp((112-60)/60)=2.38`). New weights are live.

## 3. Smoke

`artifacts/flow_ft_0830/smoke.log`: 200 steps, 9.9–10.5 it/s, resume confirmed,
sampler printed `P(n>=109)=0.636 P(n>=97)=0.872 P(n<=60)=0.006`, split
`8100 files kept, 900 dropped`, VRAM cap 46.5 GiB, ~4.7 GiB actually used.
loss 0.25–0.40 / pos_l1 0.011–0.022 (higher than round 2's console numbers purely
because the x0/hpwl terms now carry 2x weight; pos_l1 is unchanged in scale).

## 4. Live run

* Trainer PID **1614398** (uv wrapper 1614395; `setsid` detached, 8 dataloader workers).
* Log: `/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning/artifacts/flow_ft_0830/train.log`
* Ckpt dir: `artifacts/flow_ft_0830/flow_ft0830_ft250k_tailT12_lr5e-6_x02_hp06_s150k/`
* Cadence: `latest.pt` every 5k, permanent `step_XXXXXXXX.pt` every 25k,
  `keep_recent 3`, `final.pt` at 150k.
* Throughput 9.5–10.1 it/s (GPU 0 shared with another user's job) → **150k ETA
  ~2026-08-30 14:05 UTC**; worst case at 8 it/s ~15:00 UTC. Training-end deadline
  16:00 UTC, so the cosine completes with >=1 h of margin.

## 5. Commands

```bash
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
# launch / resume (bare re-run auto-resumes from latest.pt):
setsid nohup bash artifacts/flow_ft_0830/run_ft.sh > artifacts/flow_ft_0830/train.log 2>&1 &
# smoke:  bash artifacts/flow_ft_0830/run_ft.sh --max-steps 200
# rebuild seed (only for a fresh round):
#   FT_SEED_SRC=<src.pt> uv run python artifacts/flow_ft_0829/make_seed_checkpoint.py <ckdir> <lr> <warmup> <steps> <ema>
tail -f artifacts/flow_ft_0830/train.log
```
**Do not pass a different `--lr` on resume** (seed's optimizer/sched wins).
Loss-weight flags and `--warmup`/`--max-steps` DO take effect on resume.

## 6. Export the gate checkpoint (EMA-only, same as rounds 1-2)

```bash
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
CK=artifacts/flow_ft_0830/flow_ft0830_ft250k_tailT12_lr5e-6_x02_hp06_s150k
uv run python scratchpad/flow_ft_0828/export_ema_only.py \
  $CK/step_00150000.pt \
  artifacts/flow_ft_0830/flow_ft0830_tailT12_lr5e-6_x02_hp06_150k_ema.pt
# prints "... step=150000 method=flow_matching_v3 bytes=~430M" -- VERIFY step==150000.
```
`final.pt` is equivalent at 150000 but currently holds the step-200 smoke until
the run ends — prefer the numbered snapshot. A mid-run snapshot (125k) is
**un-annealed** = the §17t failure mode; only use it if the schedule slips.

Gate arm: `FLOW_CKPT=$PWD/artifacts/flow_ft_0830/flow_ft0830_tailT12_lr5e-6_x02_hp06_150k_ema.pt`
vs the round-2 arm and the packed baseline, current pack env, interleaved, >=2 reps.
**Do not run gates from this track** (GPU 3 / one evaluator at a time is the scheduler's).

## 7. Reading the gate

1. `official` raw paired delta is the clean read (identical hygiene across rounds).
2. Per-band deltas (<76 / 76-89 / 90-104 / 105-120) **and column-shipping counts
   per band** — a mid-band column jump = §17t collapse returned → reject.
3. runtime avg/max must not rise; `[selfcheck] flow_warm_latency` ~0.07 s.
4. Promote only if `official` does not regress AND shadow suites are
   sign-consistent or better; otherwise ship the 0828b pack unchanged.

## 8. Results

| tag | step | off | v3 | v5 | v6 | avg_rt | feasible | verdict |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
