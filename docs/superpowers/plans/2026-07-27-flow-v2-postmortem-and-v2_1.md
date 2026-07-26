# Flow v2 post-mortem and the v2.1 correction

Date: 2026-07-27
Branch: `codex/5.6-sol-flow-matching-f1-f3`
Author: deep-reasoner (Opus), for the Fable-5 scheduler
Supersedes: [2026-07-25-flow-v2-recipe.md](2026-07-25-flow-v2-recipe.md)

> **VERDICT: v2 lost. The SNR endpoint weighting is convicted and removed.
> v2.1 reverts the objective to v1's and keeps only the LR-horizon change.
> Expected score gain of a v2.1 retrain over v1 is ~0 — read
> "Should we retrain at all?" before spending GPU.**

---

# Part I — diagnosis

## The claim under test

v2@765k D+F+antithetic full-100 scored **1.1655** (projected 0.9133) vs v1@1M
**1.1387** (0.8855): **+0.0268 regression**. Train logs showed v2 worse on `v`
(+20-28%) and `mib` (+38-50%) from step 5k onward.

## Finding 1 — most of the logged train-curve gap is a measurement artifact

v2 draws 10% of samples from `t ∈ [0.98, 1)`. That band is not a neutral sample
of the loss surface:

- `v` (velocity MSE) **rises steeply with t** — measured 0.027 at t=0.05 to
  0.198 at t=0.99 — because the target `z0 − noise` becomes dominated by the
  unrecoverable noise term. Terminal samples therefore **inflate** logged `v`.
- The geometry losses are `t²`-gated, and — a discovery from this probe —
  **they do not vanish at the ground truth**: `mib_aspect(GT) ≈ 0.589`,
  `overlap(GT) ≈ 0.002`. So terminal samples sit at `mib`'s non-zero floor with
  near-maximal `t² ≈ 0.96` gating, **inflating** logged `mib`.

Feeding the measured `metric(t)` curves through each recipe's t-distribution
reproduces the logged ratios closely:

| logged metric | predicted v2/v1 | actual v2/v1 |
|---|---|---|
| `v`      | 1.16 | 1.22 |
| `ov`     | 1.13 | 1.14 |
| `mib`    | 1.27 | 1.39 |
| `pos_l1` | 0.96 | 0.98 |

**The "mib dilution" hypothesis is refuted as stated.** `mib`'s *weight* was
never diluted — its *log statistic* was inflated by terminal samples parked at
its ground-truth floor. Symmetrically, v2's logged "wins" on `pos_l1`/`cg`/`hp`
were artifacts hiding real regressions.

## Finding 2 — composition-corrected, the real regression is smaller and narrower

Step-matched (v1@775k vs v2@773k), same batches, same noise, v1 metric
definitions, evaluated on a fixed t-grid and re-integrated under v1's uniform
t (probe: `scripts/probes/flow_t_resolved_probe.py`, CPU-only):

| metric | TRUE v2/v1 | logged v2/v1 |
|---|---|---|
| `v`      | **1.039** | 1.22 |
| `mib`    | **1.031** | 1.39 |
| `x0`     | **1.047** | — (definitions differ) |
| `pos_l1` | **1.057** | 0.98 |
| `ov`     | **1.207** | 1.14 |

So: coordinate quality is uniformly ~4-6% worse, and **overlap quality is ~21%
worse** — the one substantial casualty. `mib`'s apparent 38% collapse was 97%
artifact.

## Finding 3 — the fingerprint convicts the SNR weighting

The x0 ratio is **monotone decreasing in t**, crossing 1.0 at t≈0.75:

| t | 0.05 | 0.15 | 0.30 | 0.45 | 0.60 | 0.75 | 0.90 | 0.97 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| x0 v2/v1 | 1.066 | 1.070 | 1.062 | 1.048 | 1.039 | 0.999 | 0.989 | 0.939 | 0.851 |

That is precisely the shape of `w(t) = clamp(t²/(1−t)², 5)`, which crosses 1 at
t=0.5 and saturates at t=0.69. Measured on the real `x0(t)` curve, the weight
moved the endpoint gradient mass:

- mass in `t < 0.5`: **64.4% → 10.1%**
- mass in `t > 0.69`: **15.6% → 56.1%**

v2 bought high-t endpoint accuracy and paid for it everywhere below t≈0.75.
Since Euler integrates through **all** t and low-t errors propagate down the
whole trajectory, that is a bad trade.

**Why the original reasoning was wrong — a category error.** In diffusion,
`x0_hat = (z_t − σ·ε̂)/α` *amplifies* the residual by `1/α` as noise rises; that
divergence is the pathology min-SNR repairs. On the straight path,
`x0_hat = z_t + (1−t)·v̂` has residual `(1−t)·(v̂ − v)` — **attenuated, never
amplified**. There was no pathology. The unweighted x0 loss was already a
well-conditioned, `(1−t)`-profiled coordinate-space auxiliary spanning the whole
path, complementing the flat velocity loss. The SNR weight destroyed it.

Corollary: the v2 doc's "effective x0 magnitude ×2.2" risk note was also wrong —
measured against the real `x0(t)` curve the magnitude ratio is **1.39x**, because
`x0(t)` shrinks with t exactly where `w(t)` grows. The monitoring advice derived
from it ("watch ov/bd/cg") pointed at symptoms, not the mechanism — and `mib`,
the metric that looked worst, was not on the list precisely because the
mechanism was misidentified.

## Finding 4 — terminal-t: not convicted, but not worth its price

It does exactly what it was designed to do: v2 is better at t≥0.97 on every
metric (`v` 0.86x, `x0` 0.85x, `pos_l1` 0.80x at t=0.99). But:

- it spends 10% of the sample budget where the endpoint is already ~10x more
  accurate than at low t (`pos_l1` 0.003 vs 0.034);
- it removes 10% of training mass from the `t²`-gated geometry losses, whose
  entire useful range is `t < 0.95` — a plausible co-contributor to the `ov`
  regression, which the SNR fingerprint does **not** explain (x0 is neutral at
  t=0.75-0.90 yet `ov` is 1.26-1.36x worse there);
- because `mib_aspect(GT) ≈ 0.589`, training at t≈0.99 under near-maximal `t²`
  gating applies pressure to push the endpoint *away* from ground truth.

The two changes were made together, so **SNR vs terminal-t cannot be fully
separated for the `ov` regression** from existing data. The x0-vs-t fingerprint
convicts SNR specifically; `ov` remains ambiguous between the two. That
ambiguity is what arm B exists to resolve.

## Finding 5 — the +0.0268 full-100 delta is real in direction, poorly measured in magnitude

Weighted per-case attribution (evaluator weighting is `exp(n/12)`, so n=120
carries ~8x the average weight):

- **id 99 (n=120) alone = 58.1%** of the regression (`hpwl_gap` 0.071 → 0.407)
- top 3 cases = 79.4%, top 5 = 96.6%
- across all 100: 51 worse, 31 better, 18 identical

Against that, a **same-checkpoint repeat evaluation** (`budget35_dm2_colcache`
vs `_rep2`) differs by **+0.0064** total, with a single case (id 89, n=110)
swinging `cost` by **+0.1619** on its own — 94% of that pair's total-score noise
variance. Per-case swings of the size that drives the v2 verdict occur from pure
evaluation stochasticity; the largest-weight case simply did not happen to swing
in the repeat pair.

**Honest position:** the direction is corroborated by the training-side evidence
(which is not noise — same seed, same init, same data order, divergence visible
from step 5k across thousands of logged steps), but the *magnitude* +0.0268 is
one draw from a heavy-tailed distribution and should not be quoted as the size
of the recipe effect. The composition-corrected training-side effect (~5%
coordinate, ~21% overlap) is the reliable number.

Grouping the per-case deltas by block-count band, MIB-violation presence, and
boundary-coded density produced **no** interpretable structure once weighted —
the regression is not "MIB-dense cases", it is "case 99 plus four others". That
is itself the result: the dilution hypothesis has no per-case support either.

## Finding 6 — the seed confound is smaller than feared, and the LR-horizon premise is dead

Both runs used `--seed 1234`: **identical initialization and identical
file-shuffle data order.** Only the RNG stream for t/noise diverges (v2 draws
extra randoms in `sample_flow_t`). So the training-curve comparison is far better
controlled than a generic "two runs, two seeds" comparison — seed variance is
largely *not* the explanation for a systematic 20%+ divergence sustained over
thousands of logged steps. (Run-to-run variance with a *different* seed remains
unmeasured for this whole channel; arm A addresses it.)

Separately, **Fix 2's premise is refuted by v1's own evidence**: v1@650k ran with
LR still at ~28% of peak, v1@1M was fully annealed to the 0.01 floor, and the
score did not improve (1.1401 → 1.1443, i.e. within noise). **Full LR annealing
is worth ~0 score for this model.** Fix 2 buys wall-clock, not quality.

---

# Part II — v2.1 recipe

## DO

1. **Remove the SNR endpoint weighting.** `--x0-time-weighting none` is the
   default; the endpoint loss is exactly v1's again. Convicted by Finding 3.
2. **Terminal-t OFF by default** (`--term-t-prob 0`, uniform t). Not convicted,
   but unproven benefit against a measured cost (Finding 4), and its original
   justification was fixing the Heun corrector — which the probe already judged
   harmful and which we are not adopting. Under the project's evidence rules an
   unproven extra does not get to be the default.
3. **Keep `--max-steps 800000`.** Harmless and saves ~20% wall-clock; but per
   Finding 6 expect no score from it.

Net: **v2.1's objective is v1's objective.** Both v2 arms survive as flags so an
ablation needs no code edit.

## NOT in v2.1

- **Both v2 loss changes as defaults** — see above.
- **Hinging `mib_aspect` at its ground-truth value** (the way `hp_loss` already
  hinges at `hp_gt`). This is a genuine, evidence-backed finding from the probe:
  `mib_aspect(GT) ≈ 0.589`, so the term pushes the model away from the ground
  truth over its whole range, not just at terminal t. It is a *promising*
  change — and precisely for that reason it does not get bundled into a
  correction release. The lesson of v2 is that stacking unvalidated changes
  makes the result uninterpretable. Ablation arm C, or v3.
- **Dynamic HPWL node feature / logit-normal t / wide high-t oversampling** —
  unchanged from the v2 doc: deferred.

## Should we retrain at all? (read before spending GPU)

Expected score gain of v2.1 over the existing v1@1M checkpoint is **~0**, because
v2.1's objective *is* v1's objective and Finding 6 shows the LR horizon buys
nothing. v1@1M (1.1387) is already the best flow checkpoint we have and is
already on disk.

Recommendation: **do not retrain flow to chase score.** Retrain only to (a) run
the ablation below as a controlled experiment, or (b) measure training
run-to-run variance, which is currently unknown for this channel and silently
underwrites every training-side decision we make. If GPU is scarce, spending it
elsewhere dominates.

## Code changes (v2.1)

- `partner/flow_matching_claude.py`: `endpoint_time_weight(t, mode, gamma)`
  dispatches `none` (identity) / `snr`; `endpoint_snr_weight` retained with its
  disproof recorded in the docstring; `sample_flow_t` default `term_prob=0.0`.
- `partner/flow_train_claude.py`: `parse_flow_extras()` lifted to module level
  (testable defaults); `--x0-time-weighting {none,snr}` (default `none`),
  `--term-t-prob` default `0.0`; tag `flow_matching_v2_1`; `checkpoint_method`
  accepts `{v1, v2, v2_1}`; default checkpoint dir
  `checkpoints/flow_matching_v2_1`.
- `scripts/probes/flow_t_resolved_probe.py`: new CPU-only diagnostic that
  produced Findings 1-4.
- `tests/test_partner_flow_training.py`: defaults locked to the v1 objective,
  weighting-mode dispatch, extras pass-through, uniform-t default.

Verification: 55 tests pass; both arms run end-to-end for real training steps on
CPU with finite losses; checkpoints round-trip the `flow_matching_v2_1` tag and
the arm settings.

---

# Part III — ablation design (200k steps/arm, ~7h each)

SNR is convicted, but terminal-t is not resolved and **training run-to-run
variance has never been measured** for this channel (v1 and v2 are each n=1).

| arm | config | question it answers |
|---|---|---|
| **A (control)** | v2.1 defaults, `--seed 4321` | Run-to-run variance: how far does a *different seed* move the curves at an identical recipe? Also validates the revert. |
| **B** | v2.1 + `--term-t-prob 0.10` | Does terminal-t help or hurt *on its own*, with SNR removed? |
| **C (optional)** | v2.1 + `mib` hinged at GT | Does removing `mib`'s 0.589 ground-truth floor help? Needs a code change first — do not run C without it. |

No arm re-tests the SNR weighting: Finding 3 is a mechanism-level disproof, and
spending 7 GPU-hours to reconfirm it is not worth it. (`--x0-time-weighting snr`
remains available if a reviewer disagrees.)

**Judgement criterion (primary):** the t-resolved probe at each arm's 200k
snapshot vs `checkpoints/flow_matching_v1/step_00200000.pt`, on identical
batches. Far more sensitive than a full-100 eval at 200k, where the model is
weak and evaluation noise is ±0.006 with the heavy tail of Finding 5.
**Arm A's spread against v1@200k is the noise floor** every other arm must clear
— if arm B's effect does not exceed arm A's spread, terminal-t is undecided, not
proven neutral.

**Judgement criterion (secondary):** D+F full-100 at the 200k snapshot — a
sanity check only, not a verdict, given Finding 5.

## Launch commands (scheduler runs these; GPU frees in ~2h)

Run from the eval/train script environment so `PYTHONPATH` reaches `partner/`.

```bash
# Arm A — control / variance floor.  Run this first: it is the only arm whose
# result is required before the others can be interpreted.
uv run python3 partner/flow_train_claude.py \
    --checkpoint-dir checkpoints/flow_v2_1_armA --max-steps 200000 --seed 4321 \
    --batch-size 12 --d-model 640 --layers 14 --heads 10 --node-feat-dim 32 \
    --lr 8e-5 --warmup 4000 --ema-decay 0.9998 --amp \
    --vram-fraction 0.85 --gpu-util-cap 0.95 \
    --x0-time-weighting none --term-t-prob 0.0

# Arm B — terminal-t in isolation
uv run python3 partner/flow_train_claude.py \
    --checkpoint-dir checkpoints/flow_v2_1_armB --max-steps 200000 --seed 4321 \
    --batch-size 12 --d-model 640 --layers 14 --heads 10 --node-feat-dim 32 \
    --lr 8e-5 --warmup 4000 --ema-decay 0.9998 --amp \
    --vram-fraction 0.85 --gpu-util-cap 0.95 \
    --x0-time-weighting none --term-t-prob 0.10 --term-band 0.02
```

`--max-steps 200000` also compresses the cosine schedule into 200k, so the arms
are internally consistent but **not** directly comparable to v1's 160-200k window
on LR. Compare arms to each other; use v1@200k for orientation only.

At the ~8.2 sps observed in the v2 run (exclusive card, `--gpu-util-cap 0.95`),
200k steps ≈ **6.8 h/arm**; A+B ≈ 14 h.

## Probe command (CPU-only, safe to run while the GPU is busy)

```bash
uv run python scripts/probes/flow_t_resolved_probe.py \
    --repo <worktree-root> \
    --ckpt v1=checkpoints/flow_matching_v1/step_00200000.pt \
           armA=checkpoints/flow_v2_1_armA/step_00200000.pt \
           armB=checkpoints/flow_v2_1_armB/step_00200000.pt \
    --batches 4 --batch-size 8 --num-samples 256 --threads 32 \
    --output artifacts/flow_ablation_tprobe.json
```

Caveat: the probe scores on the first N training samples, which both models saw,
so it measures fit rather than generalization. It is a fair *relative* comparison
(identical batches, noise, and metric definitions) and that is what the ablation
needs — but do not read its absolutes as validation metrics.
