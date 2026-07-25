# Flow Matching v2 recipe

Date: 2026-07-25
Branch: `codex/5.6-sol-flow-matching-f1-f3`
Author: deep-reasoner (Opus), for the Fable-5 scheduler

## Summary

v1 flow matching (`checkpoints/flow_matching_v1/final.pt`, 1M steps) already
**ties direct** F-only (no-runtime 1.1446 vs 1.1518) and D+F 650k=1.1401 /
1M=1.1443. Convergence evidence: **650k -> 1M gained nothing** (within
+-0.003-0.005 variance). v1 still carries three recipe defects surfaced in the
0723 review. v2 is a **recipe fix, not an architecture change**: same model,
same z-repr, same features, same sampler code. Only the *loss weighting*, the
*t-sampling*, and the *LR anneal horizon* change.

Net expected effect: better endpoint precision (the reweighting puts gradient
where the endpoint is trustworthy), a terminal region the Heun/Euler samplers
can actually evaluate, and a fully-annealed LR by the real convergence point --
all at a **shorter** 800k horizon.

## DO in v2

### Fix 1 (S-2): SNR-analogue weighting on the endpoint (x0) loss

**Defect.** On the straight path `z_t = (1-t)*noise + t*z0`, the implied
endpoint satisfies `z0_hat - z0 = (1-t) * (v_pred - v_target)`. So the
*unweighted* x0 smooth-L1 in v1 is implicitly a `(1-t)**2`-weighted velocity
error -- it is **dominated by the high-noise, low-t regime where the endpoint is
least reliable**. direct_train_v2 fixed the diffusion analogue with `w_snr =
clamp(alpha^2/sigma^2, gamma)`; the flow trainer never got it.

**Fix.** Multiply the x0 loss by the straight-path SNR analogue
`w = clamp(t^2/(1-t)^2, gamma)` (`endpoint_snr_weight`, `--min-snr-gamma`,
default 5, an already-present-but-dormant arg). Mechanism: the `t^2/(1-t)^2`
factor **cancels** the intrinsic `(1-t)^2` amplification, leaving a *net `t^2`
emphasis* on the underlying velocity error -- coherent with the geometry losses
(which already ramp by `t^2`). The `gamma` clamp mirrors direct_train_v2 and
keeps the weight finite as `t -> 1`.

**Expected effect.** Gradient re-aimed from the noise end (where `z0_hat` is
garbage) to the mid/high-t band (net emphasis peaks ~t=0.69). Should lower
`pos_l1` and tighten final coordinates.

**Risk / monitoring.** The normalizer intentionally **excludes** `w_snr`
(matching direct_train_v2), so the *effective* x0 magnitude rises ~2.2x (mean of
the clamped weight over the t distribution). This is a deliberate rebalance
toward reliable endpoints, but it shifts weight away from the geometry/HPWL
terms. **Monitor `ov`/`bd`/`cg` in the train log**; if they climb, dial
`--x0-loss-weight` below 1.0. Kept at 1.0 by default to match the proven direct
sibling config.

### Fix 2: LR anneal horizon = the real convergence point (800k)

**Defect.** The cosine denominator is `max_steps - warmup`. v1 ran with
`--max-steps 1000000` but converged by ~650k, so at 650k the LR was still ~28%
of peak and the final 350k ran at low LR with no gain.

**Fix.** Run v2 with `--max-steps 800000` (the flow default) and **do not
override it**. Cosine now fully anneals to the 0.01 floor by 800k, spending the
budget where it moves the loss. No code change beyond the default. 800k (not
650k) leaves a modest margin above the observed convergence knee.

### Fix 3: terminal-t coverage for the Heun/Euler endpoint

**Defect.** Uniform `t = torch.rand` never reaches 1.0, but the Heun corrector
evaluates the model at exactly `t1 = 1.0` (and Euler's final substep sits just
below it). The near-data regime -- where final placement precision is set -- is
also under-sampled. The probe found **Heun harmful** on v1.

**Fix.** `sample_flow_t` draws a fraction `--term-t-prob` (default 0.10) of
samples from the endpoint band `[1 - term_band, 1)` (`--term-band`, default
0.02). This trains the velocity field in the terminal neighbourhood (fixing
Heun's weak point and sharpening Euler's last step) with **reducible targets**.
A hard `t = 1.0` spike is deliberately avoided: at t=1 the target `z0 - noise` is
unpredictable from the clean state and carries an irreducible loss floor; the
0.02 band gives the same time-embedding coverage without that floor.

**Sampler stance.** Keep **Euler as the recommended inference sampler**; the
sampler code is untouched. Terminal coverage merely gives Heun a fair re-test:
after v2 trains, re-run `scripts/probes/flow_candidate_probe.py` euler-vs-heun.
Do not adopt Heun without fresh probe evidence.

**Ablation knob.** `--term-t-prob 0` recovers exactly v1's uniform t.

### Fix 4: geometry `t^2` gating -- reviewed, KEPT

The geometry/HPWL losses (`ov/bd/cg/mb/hp`) keep their `endpoint_weight = t^2`.
The `(1-t)^2`-cancellation algebra of Fix 1 is specific to the L2-like x0
residual; the geometric penalties are non-linear functions of the decoded rects,
where `t^2` is simply the correct "trust the endpoint more as t -> 1" ramp (max
weight at t=1, exactly where those penalties become meaningful). Since v1's
geometry weighting already tied direct, changing it would be an unvalidated
rebalance. x0 (SNR) and geometry (t^2) both emphasize the reliable high-t band;
they do not conflict.

## NOT in v2 (deferred, with reasons)

- **Dynamic HPWL node feature (MacroDiff+ Table 4, item 5) -> v3.** Feeding the
  current noisy-state net-HPWL back into `node_feat` each step changes the model
  input contract: `node_feat_dim` grows, `fast_condition`/`cond` assembly and
  the sampler must both recompute it identically every step, and any mismatch is
  train/inference skew. That is an **architecture change with real skew
  surface** and uncertain ROI in this setup -- out of scope for a recipe fix.
  Revisit in v3 with a dedicated skew test.

- **Logit-normal / mid-focused t distribution (SD3, item 6) -> not now.** SD3's
  mid-t focus targets image-diffusion's hardest transition. Our auxiliary
  objective wants the **data end**, not the middle; Fix 3 already biases toward
  the endpoint. Adding an unvalidated distributional shift on top would confound
  the v1->v2 read. If v2 underperforms, a data-end oversampling A/B (below) is
  the cleaner next lever than mid-focus.

- **Wide high-t band oversampling (the direct_v2 low-t-oversampling port).**
  Genuinely tempting -- it is the single biggest structural gap vs direct_v2
  (flow-v1 uses plain uniform t) -- but there is **no flow-specific evidence**
  it helps, and over-weighting high t can starve the low-t field that Euler's
  early steps integrate through. Given one expensive 800k run, the conservative
  default (uniform base + 10% terminal band) is the right bet. Exposed as a
  future A/B: widen `--term-band` toward ~0.25 and raise `--term-t-prob`, then
  compare on the probe before committing.

## Code changes

- `partner/flow_matching_claude.py`: new pure helpers
  `endpoint_snr_weight(t, gamma)` and
  `sample_flow_t(n, device, generator, term_prob, term_band)`.
- `partner/flow_train_claude.py`:
  - `checkpoint_method` now accepts the family `{flow_matching_v1,
    flow_matching_v2}` (v1 must stay loadable for `flow_candidate_probe.py`);
    training tags checkpoints `flow_matching_v2`.
  - `flow_train_step`: t drawn via `sample_flow_t`; x0 loss multiplied by
    `endpoint_snr_weight`.
  - default `--checkpoint-dir checkpoints/flow_matching_v2`; new
    `--term-t-prob` (0.10) / `--term-band` (0.02) flags.
- `tests/test_partner_flow_training.py`: v2 acceptance + helper unit tests
  (SNR-weight monotonicity/clamp, terminal-band fraction). `test_partner_*`
  suites green (49 passed).

## Launch command (scheduler runs this; do not start until the GPU eval frees)

```bash
cd FloorSet/iccad2026contest   # via the eval/train script env, or set PYTHONPATH to partner/
uv run python3 ../../partner/flow_train_claude.py \
    --checkpoint-dir checkpoints/flow_matching_v2 \
    --max-steps 800000 \
    --batch-size 12 --d-model 640 --layers 14 --heads 10 \
    --node-feat-dim 32 --lr 8e-5 --warmup 4000 --ema-decay 0.9998 --amp \
    --vram-fraction 0.85 --gpu-util-cap 0.95 \
    --term-t-prob 0.10 --term-band 0.02 --min-snr-gamma 5.0
```

Everything except the three fixes is inherited from v1. `--term-t-prob 0`
reproduces v1's t-sampling for a clean ablation.

### Estimated wall time

v1 sps was **5.7** on a *shared* card at `--gpu-util-cap 0.75` (duty-cycle
sleep of `1/0.75 - 1 = 0.33x` busy per step). Removing most of that throttle
(0.95 -> `0.05x`) alone lifts wall sps ~1.27x; an **exclusive** card removes
contention on top. Realistic range **~8-14 sps**.

- at 8 sps: 800k / 8 = 100k s ~= **28 h**
- at 11 sps: ~20 h
- at 14 sps: 800k / 14 = 57k s ~= **16 h**

Plan for **~24 h**; confirm sps from the first log lines and adjust. The
convergence knee is ~650k, so **evaluate the 650k snapshot and final.pt (800k)**
and promote whichever wins on full-100 no-runtime (the 650k point lands
~13-18 h in). Snapshots are saved every `--snapshot-every` (25k) steps.
```
