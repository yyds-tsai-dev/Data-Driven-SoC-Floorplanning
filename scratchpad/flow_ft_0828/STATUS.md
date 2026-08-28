# Flow fine-tune 0828 — model-training track STATUS

Owner: deep-reasoner (model-training track). Branch `final-sprint-0827`. **No commits.**
Last update: 2026-08-28 ~02:20 local. **Training is RUNNING (detached).**

---

## 0. TL;DR for the next session

* A tail-tilted fine-tune of the shipped Flow v1 prior is training on **GPU 2**,
  detached (`setsid nohup`), PID group parented at `artifacts/flow_ft_0828/run_ft.sh`.
* Log: `/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning/artifacts/flow_ft_0828/train.log`
* Checkpoints: `artifacts/flow_ft_0828/flow_ft0828_ftv1_tailT24_w0-89_lr2e-5_wu1k_bs12_s300k/`
  (`step_XXXXXXXX.pt` every 5k, permanent snapshots every 25k, `latest.pt`).
* Throughput ~**12 it/s** (7.6–17 range, box load dependent) → 25k steps ≈ 35 min,
  300k steps ≈ 7 h.
* **No gate has been run yet.** Gate the 25k snapshot first (sanity), then ~100k,
  then the annealed 300k final. Procedure in §6.

---

## 1. Reconstructed v1 recipe (evidence, not guesswork)

Source: `torch.load('submission/cadc1013/checkpoints/flow_matching_v1_final.pt',
weights_only=False)` → keys `['model','ema','model_config','args','step']`, `step=1000000`.

`model_config`:
```
node_feat_dim=32, relation_feat_dim=9, z_dim=4, z_repr='xyaspect',
d_model=640, layers=14, heads=10, dropout=0.0, timesteps=1000,
self_conditioning=True                       # 107.5M params
```

`args` (verbatim):
```
training_method='flow_matching_v1', data_path='FloorSet',
checkpoint_dir='checkpoints/flow_matching_v1', fresh=True, resume=None,
batch_size=12, grad_accum_steps=1, num_samples=None, max_steps=1000000,
lr=8e-05, warmup=4000, ema_decay=0.9998, amp=True, device='cuda',
num_workers=4, seed=1234, augment=1, file_shuffle=1,
self_cond_prob=0.5, min_snr_gamma=5.0, bnd_node_weight=2.0,
v_loss_weight=0.5, x0_loss_weight=1.0, overlap_loss_weight=0.5,
boundary_loss_weight=0.3, cluster_loss_weight=0.3, mib_loss_weight=0.1,
hpwl_loss_weight=0.3,
vram_fraction=0.45, gpu_util_cap=0.75,
save_every=1000, snapshot_every=25000, keep_recent=3, log_every=50
```

Notes that matter:

* v1's `args` has **no** `x0_time_weighting` / `term_t_prob` / `term_band` /
  `mib_hinge` keys — it predates those flags. Today's
  `partner/flow_matching_train.py` defaults (`x0_time_weighting='none'`,
  `term_t_prob=0.0`, `mib_hinge` off) reproduce the v1 objective **bit-exactly**,
  so a fine-tune with those defaults is a pure continuation.
  (`TRAINING_METHOD` is now `'flow_matching_v3'`; that tag is in `FLOW_METHODS`,
  so `contest_optimizer._load_flow_model` accepts the resulting checkpoint.)
* Docs (`docs/superpowers/plans/2026-07-25-flow-v2-recipe.md`,
  `2026-07-27-flow-v2-postmortem-and-v2_1.md`) record: v1 ran 5.7 it/s on a
  *shared* L4 at `--gpu-util-cap 0.75`; **v1 converged by ~650k — 650k→1M gained
  nothing**; v2 (SNR endpoint weighting) regressed the suite by +0.0268 and was
  reverted; **no fine-tune-from-checkpoint run and no tail/block-count-weighted
  sampling experiment has ever been done for this model.**
* Evaluator noise floor (postmortem Finding 5): a same-checkpoint repeat eval
  moved the total score by **0.0064**, 94% of it from one n=110 case. Deltas
  below ~0.005 are noise; pair and repeat.

## 2. Measured throughput (GPU 2, H100 NVL)

200-step smoke, tail-tilted sampler, batch 12, `--num-workers 8`,
`--gpu-util-cap 0` (throttle off — GPU 2 is ours), `--vram-fraction 0.5`:

```
13.7 – 17.9 it/s (fresh weights), 4.7 GiB GPU memory in the live run
```
Live run steady state: **7.6 – 17 it/s, ~12 it/s average** (box load average was
25–30 during the smoke; the swing tracks CPU contention, i.e. the dataloader,
not the GPU). 8 workers is enough that the GPU is not starved at the top of the
range; the low excursions coincide with load spikes from the other users.
This is ~2.1× v1's 5.7 it/s, mostly from disabling the duty-cycle throttle.

## 3. The fine-tune design and why

**Problem.** The pipeline's residual is a tail problem: HPWL gap ≈0.02 (public) /
0.03–0.04 (v3/v5/v6) at n≥102, and rung-0 (fixed frame at 1.02·area_ref) closes
only ~17% of ladder attempts.

**The largest, and completely untried, train/metric misalignment is the
block-count distribution.**

* The official score is `Σ cost_i·e^{(n_i-n_max)/12} / Σ e^{(n_j-n_max)/12}`
  (`FloorSet/iccad2026contest/iccad2026_evaluate.py:581-603`). With n_max=120:
  **63.2% of the score mass sits at n≥109, 86.5% at n≥97, 95.0% at n≥85.**
* The FloorSet-Lite training corpus is near-uniform in n. On the train split,
  `P(n≥109)=0.119`, `P(n≥97)=0.240`, `P(n≤60)=0.398`.
* v1's only tail emphasis is `large_case_weight = exp((n-60)/60).clamp(0.5,3)`
  (`partner/flow_matching_train.py`), a **batch-scalar** multiplying the whole
  loss. Because `FileShuffleSampler` walks one 112-layout file at a time and every
  file is n-homogeneous, batches are n-homogeneous too, so this acts as a per-n
  LR modulation with a total dynamic range of only **5.2×** (n=21→0.52,
  n=120→2.72) against a score-weight range of **3800×**.

**Chosen change (one variable): tempered tail-tilted file sampling + LR restart.**

* Files are drawn **with replacement** ∝ `exp((n-120)/24)`.
  T=24 is the *square root* of the official weight `exp((n-120)/12)` — the
  classic variance-safe compromise between the training distribution (ignores
  the metric) and full importance sampling (collapses support, and would
  cycle only ~1000 tail files).
  Resulting step shares: `P(n≥109)=0.402` (vs 0.119 uniform, 0.632 score mass),
  `P(n≥97)=0.648`, `P(n≤60)=0.067`. Per-n multipliers: n=120 **4.17×**,
  n=90 1.20×, n=21 0.067×.
* Oversampling, not loss reweighting: with n-homogeneous batches a loss weight is
  just a per-step LR spike (high gradient-noise); oversampling adds *steps* at
  constant gradient scale.
* **The objective is left bit-exact to v1** (all loss weights unchanged,
  `x0_time_weighting=none`, `term_t_prob=0`, `mib_hinge` off). This is deliberate:
  gate slots are scarce and the v2 postmortem shows objective surgery is how this
  model got worse before. One variable.
* LR: peak **2e-5** (25% of v1's 8e-5), warmup 1000, cosine → 0.01× over 300k.
  v1 finished its own cosine at 8e-7, so a restart is required for any movement.
  Note the pre-existing `large_case_weight` means tail steps run at ≈2.7×
  effective LR (~5.4e-5), which is why the peak is a quarter of v1's, not half.
* EMA decay 0.9998, identical to v1 (half-life ≈3.5k steps — fully re-tracks
  within the budget). The gate reads the `ema` tensors.
* **Why not "more steps"** (option d): the docs record 650k→1M gained nothing on
  the *current* distribution. Steps only help if the distribution changes.
* **Why not the HPWL/mib knobs now** (option c): `hp_loss` and the overlap term
  trade against each other (tighter wirelength → harder to legalize into the
  1.02 frame → worse rung-0 → area up), sign unknown, and confounding it with the
  sampling change would make an ambiguous gate uninterpretable. It is the
  designated **second** arm if the tail tilt gates positive and time remains.

**Risk containment.** Flow candidates only displace the column-SA champion if
they beat it by a proxy margin (`partner/contest_optimizer.py:149`), so a tilt
that costs small-n quality is bounded below by the column baseline and costs no
runtime (the same number of refine calls happen either way). This is what makes
an aggressive tail tilt cheap to try.

## 4. Data hygiene — how held-out rows were excluded

* Every shadow manifest (`shadow_hidden/shadow_hidden_{v1,v3,v5,v6}_share/manifest.json`)
  records `"workers": [90..99]` and a `source_file` under `worker_90..worker_99`
  for all 100 cases ("FloorSet-native late-worker holdout", candidate pool
  100,800 = 900 files × 112).
* The fine-tune therefore trains on **`worker_0 … worker_89` only**:
  8100 of 9000 files, **907,200 of 1,008,000 rows**. The entire shadow candidate
  pool is removed — not just the 400 selected rows.
  Enforced in `TrainSplitLite` (`artifacts/flow_ft_0828/ft_launch.py`), and the
  run log's first lines confirm it:
  `[ft] train split: 8100 files kept, 900 dropped (worker_0..worker_89); 907200 rows`.
* Public validation (`LiteTensorDataTest`) is a different container and is
  unreachable from `FloorplanDatasetLite`; it is never trained on.
* **Caveat to carry into the gate reading:** v1 itself was trained on the *full*
  corpus (`data_path='FloorSet'`, no exclusion), i.e. **v1 saw the shadow rows and
  the candidate is clean**. That biases the v3/v5/v6 paired comparison *against*
  the candidate. The `off` (public LiteTensorDataTest) arm is clean for both and
  is the unbiased read.

## 5. Exact commands (resumable)

Relaunching the script auto-resumes from `latest.pt`; nothing else is needed.

```bash
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
setsid nohup bash artifacts/flow_ft_0828/run_ft.sh > artifacts/flow_ft_0828/train.log 2>&1 &
```

`artifacts/flow_ft_0828/run_ft.sh` sets
`PYTHONPATH=FloorSet/iccad2026contest:FloorSet:partner`, `CUDA_VISIBLE_DEVICES=2`,
`FT_TAIL_TEMP=24`, `FT_MAX_WORKER=89`,
`FT_N_INDEX=artifacts/flow_ft_0828/file_n_index.json`, and runs:

```
uv run python artifacts/flow_ft_0828/ft_launch.py \
  --data-path $ROOT/FloorSet \
  --checkpoint-dir $ROOT/artifacts/flow_ft_0828/flow_ft0828_ftv1_tailT24_w0-89_lr2e-5_wu1k_bs12_s300k \
  --amp --batch-size 12 --num-workers 8 \
  --lr 2e-5 --warmup 1000 --max-steps 300000 --ema-decay 0.9998 \
  --d-model 640 --layers 14 --heads 10 --node-feat-dim 32 \
  --save-every 5000 --snapshot-every 25000 --keep-recent 3 \
  --gpu-util-cap 0 --vram-fraction 0.5 \
  --hpwl-loss-weight 0.30 --x0-time-weighting none --term-t-prob 0.0
```

**Do not change `--lr`** on a resume: the seed's `optimizer`/`sched` state carries
`base_lrs=[2e-5]`, and `LambdaLR.load_state_dict` overwrites the CLI value.
`--warmup` / `--max-steps` DO take effect on resume (the lambda is rebuilt from
the CLI; only `last_epoch` is restored).

### Files written (all inside artifacts/ and scratchpad/ — no shipped file touched)

| path | what |
|---|---|
| `artifacts/flow_ft_0828/ft_launch.py` | patches `direct_diffusion_train.FloorplanDatasetLite` → train-split-only, and `.FileShuffleSampler` → tail-tilted sampler, then calls `flow_matching_train.main()`. Trainer code itself is untouched. |
| `artifacts/flow_ft_0828/make_seed_checkpoint.py` | v1's stripped ckpt has no `optimizer`/`sched`, which `main()` requires on resume. Builds `latest.pt` at step 0 with v1's `model`+`ema` and fresh AdamW/LambdaLR state. |
| `artifacts/flow_ft_0828/build_file_n_index.py` + `file_n_index.json` | {relpath → n} for all 9000 lite files (dim-1 of the input tensor is n; no intra-file padding). n∈[21,120], 65–112 files per n, 0 failures. |
| `artifacts/flow_ft_0828/export_gate_ckpt.py` | strips a `step_*.pt` to the shipped `{model,ema,model_config,args,step}` form for gating. |
| `artifacts/flow_ft_0828/run_ft.sh` | the detached launcher above. |

Resume sanity check: at step ~350 the loss was **0.138 / v 0.044 / pos_l1 0.0096**;
a from-scratch run at the same step sits at **2.6 / v 1.27 / pos_l1 0.73**. v1's
weights are loaded.

## 6. Gating procedure (NOT yet started)

Only when `pgrep -f iccad2026_evaluate.py` is empty (another chain owned GPU 3 as
of 02:07; poll before starting).

```bash
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
CK=artifacts/flow_ft_0828/flow_ft0828_ftv1_tailT24_w0-89_lr2e-5_wu1k_bs12_s300k
uv run python artifacts/flow_ft_0828/export_gate_ckpt.py $CK/step_00025000.pt
# -> artifacts/flow_ft_0828/flow_ft0828_tailT24_lr2e-5_step25k.pt

G=/ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/run_gate5.sh
COMMON="PARTNER_COORD_POLISH= PARTNER_BUDGET_TABLE=$(cat artifacts/p0_newbox/budget_table_mid.txt) \
DIRECT_OFF= DIRECT_CKPT=$PWD/artifacts/icdc_topology/checkpoints_s2_20k/best.pt \
PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10 PARTNER_REFINE_RES_FRAC=0.45 \
PARTNER_WALL_REPAIR=1 PARTNER_FLOW_WARM=1 PARTNER_EARLY_EXIT=1 \
PARTNER_REFINE_SECURE_FALLBACK=1 PARTNER_SEAT_R0_ADAPT=0 CUDA_VISIBLE_DEVICES=3"
# alternate arm order between repetitions; >=2 reps each arm
bash $G ftBase_r1 $COMMON FLOW_CKPT=$PWD/submission/cadc1013/checkpoints/flow_matching_v1_final.pt
bash $G ftT24_r1  $COMMON FLOW_CKPT=$PWD/artifacts/flow_ft_0828/flow_ft0828_tailT24_lr2e-5_step25k.pt
bash $G ftT24_r2  $COMMON FLOW_CKPT=...   # candidate first this time
bash $G ftBase_r2 $COMMON FLOW_CKPT=...v1...
uv run python /ldaphome/yyds-tsai-dev/.claude/jobs/06af0e53/tmp/analyze_pairs.py \
  artifacts/shadow/ftBase_r1_off.json artifacts/shadow/ftBase_r2_off.json -- \
  artifacts/shadow/ftT24_r1_off.json  artifacts/shadow/ftT24_r2_off.json
```
A GATE5 sweep takes ≈5–6 min (≈1 min/suite). Also check
`[selfcheck] flow_warm_latency` in the eval output — it must stay ≈0.07 s
(architecture is unchanged, so any drift means something else moved).

**Promotion criterion:** public (`off`) **and** v3 **and** v5/v6 improve, paired
and sign-consistent; runtime not up; 100/100 feasible.
**Stop rule:** if the 25k gate shows a clear regression on `off`, kill the run
(`pkill -f ft_launch.py`) — the model is being damaged, not tilted; v1 remains
the ship candidate.

## 7. Gate results

_(none yet — 25k snapshot ETA ≈02:50 local)_

| tag | step | off | v3 | v5 | v6 | a1 | mean4 | avg_rt | feasible | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

## 8. Open items / next actions

1. Wait for `step_00025000.pt` (~35 min from 02:15), export, gate vs v1 base, ×2 paired.
2. If sign-consistent improvement or neutral-with-tail-gain: keep training; gate
   ~100k, then the annealed 300k final.
3. If neutral at 100k: consider the second arm (`--hpwl-loss-weight 0.5`, or
   `--mib-hinge`) as a *separate* fine-tune from v1, not stacked.
4. Deadline: gated checkpoint must exist by 2026-08-30 evening; submission
   2026-08-31 23:59. The 300k run finishes ≈09:30 on 08-28, leaving ample gate time.
