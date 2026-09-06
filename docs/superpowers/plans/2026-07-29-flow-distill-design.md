# Flow-channel few-step distillation (SCFM) — design

**Date** 2026-07-29 · **Branch** `5.6-sol-flow-matching` · **Status** design + script landed, unrun
**Primary source** Cai et al., *Shortcutting Pre-trained Flow Matching Diffusion Models is Almost
Free Lunch* (SCFM), arXiv:2510.17858, NeurIPS 2025 · local copy `artifacts/papers/2510.17858.pdf`
**Deliverables** `src/solver/flow_distill_claude.py` (trainer), `tests/test_partner_flow_distill.py`
(Gate-1 invariants), `scripts/probes/flow_step_curve_probe.py` (step-count quality curve),
one-line tag registration in `src/solver/flow_train_claude.py`, this document (method + gates)

---

## 0. Headline, before the method

Two measurements taken while building this (both reproducible, scripts in §7) reframe the project:

**M1 — the teacher is already nearly straight in placement, and not at all straight in overlap.**
Undistilled `flow_matching_v1/final.pt`, deployment sampler contract (Euler, hard-anchor
imposition, self-conditioning), 64 samples, via `scripts/probes/flow_step_curve_probe.py`:

| sampler steps | position L1 vs golden | raw overlap | pos vs st8 | overlap vs st8 |
|---:|---:|---:|---:|---:|
| 1  | 0.02040 | 0.03672 | **1.15×** | **8.25×** |
| 2  | 0.01867 | 0.01413 | 1.05× | 3.18× |
| 4  | 0.01705 | 0.00680 | 0.96× | 1.53× |
| 8  | 0.01721 | 0.00481 | 1.00× | 1.00× |
| 16 | 0.01777 | 0.00445 | 1.03× | 0.93× |

Blocks land in roughly the right place after **one** Euler step. What the last steps buy is
*fine-grained separation*. So the distillation objective is not "recover the layout", it is
"recover the block separation", and that is what the auxiliary arm in §4 targets.

**M2 — the payoff is runtime, not score.** Best case, a distilled student *ties* the teacher.
Its value is entirely the freed NFE: production runs `PARTNER_FLOW_SLOTS=10 × PARTNER_FLOW_STEPS=8`
= 80 network evaluations per case; st1 makes that 10. Per the 0723 pivot ("減時 is now a
first-class lever"), that time goes back to the legalizer/SA budget.

**Consequence — run the free baseline first.** Because the column/SA legalizer repairs overlap
*by construction*, the 8.25× raw-overlap penalty at st1 may not reach the terminal score at all.
The repo's own "自洽性 > 成分品質" law says raw-metric *improvements* often fail to transfer; it
cuts the same way for raw-metric *regressions*. Two evaluator runs with the **unmodified** teacher
at `PARTNER_FLOW_STEPS=2` and `=1` cost zero training and can settle the whole question (§6, Gate 0).
Do not spend distillation GPU-hours before Gate 0 reports.

---

## 1. The method, exactly

### 1.1 Setup and the convention flip

Paper: `x_t = (1-t)x_0 + t x_1`, `x_1 ~ N(0,I)` — **t = 1 is noise**, sampling integrates t downward,
`x_{t_{i+1}} = x_{t_i} - (t_i - t_{i+1}) V(x_{t_i}, t_i)` (Eq. 1–5).

This repo (`flow_matching_claude.flow_path`): `z_t = (1-t)·noise + t·z0`, `v = z0 - noise` —
**t = 0 is noise**, `sample_flow` integrates t upward, `z_{t+d} = z_t + d·v`.

So `t_repo = 1 - t_paper`, and the velocity flips sign together with the step direction. Every
formula below is stated in **repo convention**; the sign flip cancels inside the convex combination,
so the target formula is transported unchanged. This is the single easiest place to introduce a
silent bug, and it is why `scfm_target` is a separate pure function with an identity test (§7).

### 1.2 Velocity-space self-consistency (Eq. 10–12)

Take three grid times `t1 < t2 < t3`, with `d1 = t2 - t1`, `d2 = t3 - t2`. Euler composition says
one jump of length `d1+d2` equals two consecutive jumps, which in velocity space is

```
(d1 + d2)·V(z_t1, t1)  ≈  d1·V(z_t1, t1) + d2·V(z_t2, t2)          (Eq. 11)
z_t2 = z_t1 + d1·V(z_t1, t1)
```

giving the **distillation target** (Eq. 12)

```
V_target(z_t1, t1) = d1/(d1+d2) · V_near(z_t1, t1) + d2/(d1+d2) · V_far(z_t2, t2)
```

The left side is the *student*; the right side is *stopgrad*. Because the network has **no step-size
input**, the same `V_θ(z_t1, t1)` must simultaneously satisfy this for every skip length — and the
only field that can is one whose velocity is constant along the path, i.e. a straight trajectory.
That is the entire trick: shortcut models (Frans et al., ICLR 2025) buy step-size flexibility with a
`d` embedding and a from-scratch retrain; SCFM buys it by *forcing step-size independence* on a
frozen architecture. Nothing about it is Flux-specific.

### 1.3 Who computes `V_near` and `V_far` (Eq. 13, 21, 22)

The batch of size `N` is split. For the first `k` elements (`k/N = 0.4` in the paper) the target is
anchored on the **frozen teacher**; the remaining `N-k` are **self-distilled**:

```
L_scfm = 1/N [ Σ_{i≤k} (V_θ(z_t1,t1) − V_{θ*}(z_t1,t1))²  +  Σ_{i>k} (V_θ(z_t1,t1) − V_{θ⁻}(z_t1,t1))² ]
```

with the two targets built from Eq. 12 as

| slice | grid skip | `V_near` | `V_far` (at `z_t2`) | source |
|---|---|---|---|---|
| teacher-anchored (`k`) | fixed **1** | frozen teacher `θ*` | **slow** EMA `θ⁻` | Eq. 21 ("vanilla-mix", App. C) |
| self-distilled (`N−k`) | random from `{2,4,…,T/4}` | **fast** EMA `θ⁺` (μ=0.99) | **slow** EMA `θ⁻` (μ=0.999) | Eq. 22 (App. E, **the final algorithm**) |

The Euler sub-step to `z_t2` is taken with `V_near`, so the far velocity is always evaluated at the
state the near model actually produced (App. B: "consistent solvers for both steps", then mixed).

Remark 1 of the paper is the intuition: the first term keeps the student pinned to the *teacher's
direction* at fine granularity; the second term straightens the trajectory across coarse scales.
Few-step ability at counts never trained directly (1-step) is **emergent** from step-size
independence — that is the load-bearing assumption, and the main risk (§8).

Paper's App. C/D/E ablation order: vanilla (both from `θ⁻`) < vanilla-mix (Eq. 21) < cyclic EMA
restart < **dual fast-slow EMA (Eq. 22)**, which converges fastest *and* avoids the late-stage
degradation that aggressive cyclic restarts (cyc-500) cause. We implement the winner directly and
skip cyclic restarting entirely.

### 1.4 Step schedule (Eq. 17)

The paper discretizes `L_n = Linspace(1,0,n+1)` and applies a shift `S_s(t) = st/(1+(s-1)t)` with
`s ~ U[2.5,4.5]`, concentrating steps in the high-noise region — because that is how Flux/SD3 are
*sampled*. `apply_shift` implements it in our time direction, but **defaults to off** (§3, D2).

### 1.5 Training details (§4.1, §5.1)

AdamW, lr `2e-5`, batch 16, `k/N = 0.4`, EMA μ 0.99/0.999, plain `ℓ2` in velocity space (App. B
argues explicitly against sample-space losses: they are 10–100× smaller numerically and force
lr ~1e-6 and thousands of GPU-hours). LoRA is used *only* to make a 12B model trainable and is
irrelevant at our 107.5M. Few-shot: 10 images match full-dataset results on CLIP, with modest loss
of teacher-preservation fidelity (App. F).

---

## 2. Why SCFM over the alternatives

| | progressive distillation (Salimans & Ho) | shortcut models (Frans et al.) | **SCFM** |
|---|---|---|---|
| architecture change | none | **needs a `d` embedding** | none |
| retrain from scratch | no | **yes** | no |
| stages | `log2(n)` sequential stages, error compounds | single | **single, end-to-end** |
| loss space | sample space | velocity | velocity |
| our blocker | 3 stages × full runs, each needs its own convergence call | new `model_config` field ⇒ `_load_flow_model` / probe / `DirectModelConfig` all change | — |

Shortcut models are ruled out on integration cost alone: a new conditioning input changes
`DirectModelConfig`, and every loader in `my_opt_claude._load_flow_model`,
`scripts/probes/flow_candidate_probe.py` and `flow_train_claude.checkpoint_method` would need to
agree on it, days before the deadline. Progressive distillation survives as fallback F1 (§9).

Domain corroboration already on file: CO diffusion progressive distillation, 16× acceleration for
−0.019% quality (arXiv 2308.06644); rectified-flow trajectories are near-straight by construction —
and M1 above confirms that empirically for *our* teacher.

---

## 3. Adaptation points and deliberate deviations

**A1 — Teacher = EMA weights.** Both `my_opt_claude._load_flow_model`
(`ckpt.get("ema") or ckpt["model"]`) and the candidate probe sample the EMA. Distilling the raw
weights would distil something nobody deploys. Student is initialized from the same EMA weights.

**A2 — Self-conditioning.** Our `DirectDenoiser` takes `self_cond` and `sample_flow` feeds the
previous step's implied endpoint. Handled by mirroring the sampler exactly: the inner rollout's
first call gets `self_cond=None` (as `sample_flow` does at step 0), the second call at `t2` gets
`endpoint_from_velocity(z_t1, v_near, t1)` with channel-2 clamping and hard-anchor override. The
**student's** LHS gets `None` by default, which is precisely the 1-step deployment condition and the
first call of any n-step rollout. `--student-self-cond 1` feeds it the stopgrad endpoint instead
(consistent on both sides of the loss); this is the honest gap for the 2-step arm, left as an
ablation rather than a guess.

**A3 — Hard-anchor imposition.** `--impose-known 1` (default) overwrites preplaced/fixed z channels
in `z_t2` with the analytic path value `(1-t2)·noise + t2·z_known`, matching what `sample_flow` does
at every step. Cheap and exact, because `z_t1` was itself built from `noise`.

**A4 — No CFG.** Eq. 16 is dropped; our conditioning is a graph, always present, never dropped.

**A5 — No LoRA.** 107.5M params, full fine-tune. Six weight copies (student, teacher, fast/slow EMA
shadows, fast/slow materialized) ≈ 2.6 GB + optimizer 0.9 GB + grads 0.4 GB ≈ **3.9 GB**, fits an
L4 alongside other work at `--vram-fraction 0.5`.

**D1 — Grid resolution decoupled from teacher step count.** The paper sets `L_n` to the teacher's
own `n=32`. Our teacher deploys at 8. Using `--grid-steps 32` with teacher-skip 1 means the anchor
term evaluates the teacher at 1/32 granularity — *more* accurate ODE integration of the same
continuous field, never less. Justified; the teacher is a continuous velocity field, `t_model` is a
sinusoidal embedding of a real number, and nothing is quantized to 8.

**D2 — Shift off by default** (`--shift-min/--shift-max = 1.0`). The shift is correct only if the
teacher is *sampled* with it; ours is sampled uniformly. `--grid-jitter 1` (default) replaces the
shift's role as a continuous-`t` randomizer by offsetting the whole grid by a sub-cell phase, which
is coverage without a distribution change. Both are flags, so the paper-faithful arm is one
argument away.

**D3 — Skip ladder top `grid/2` when `--steps-target 1`.** The paper caps at `T/4`. At `grid/2` the
only admissible window is `t1=0, t2=1/2, t3=1` — exactly the 2-step→1-step consistency at the
sampler's *entry point*. Our ladder is shorter than the paper's (8-step vs 32-step teacher), so the
emergent extrapolation to 1 step has less room; supervising the entry-point window directly is a
small, principled extension. `--max-skip` overrides it back to `grid/4` for the paper-faithful arm.

**D4 — Arm B auxiliaries, default OFF.** Justified by M1 (the deficit is overlap, not position):
`--overlap-aux-weight` adds `V1.overlap_fraction` on the student's implied endpoint,
`--aux-weight` adds golden-endpoint smooth-L1 as a coarse anchor. Unlike `flow_train_step`, the
default time weight is **flat**, not `t²` — a 1-step student is read out at `t → 0`, exactly where a
`t²` gate is zero. `--aux-time-weight t2` restores the training-time behaviour.
Scale note: at these settings velocity loss ≈ 8e-4 while the unweighted overlap term ≈ 2.6e-2, so
`--overlap-aux-weight 0.02` puts them near parity; 1.0 would drown the distillation objective.

---

## 4. Hyperparameters

| flag | default | paper | note |
|---|---|---|---|
| `--teacher-checkpoint` | — | — | v1 `final.pt` or the v3 full-100 gate winner; validated by `checkpoint_method` |
| `--steps-target` | 1 | — | sets the ladder top only; SCFM has no step conditioning, so one student serves all counts |
| `--grid-steps` | 32 | 32 (= teacher steps) | D1 |
| `--max-skip` | `grid/2` (st1) / `grid/4` (st2+) | `T/4` | D3 |
| `--teacher-frac` | 0.40 | 0.4 | `k/N` in Eq. 13 |
| `--ema-fast` / `--ema-slow` | 0.99 / 0.999 | 0.99 / 0.999 | Eq. 22. **Not** the trainer's 0.9998 — that is a 5000-step horizon, far too slow for a ≤20k-step run |
| `--lr` | 2e-5 | 2e-5 | velocity-space `ℓ2` tolerates it; sample-space would need ~1e-6 (App. B) |
| `--batch-size` | 12 | 16 | matches the v3 run's L4 footprint |
| `--warmup` / `--max-steps` | 200 / 20000 | — | paper: 8-step converges ~1000 it, 3-step < 24 A100-h on 12B |
| `--grid-jitter` | 1 | (shift) | D2 |
| `--shift-min/max` | 1.0 / 1.0 | U[2.5,4.5] | D2 |
| `--impose-known` | 1 | n/a | A3 |
| `--student-self-cond` | 0 | n/a | A2 |
| `--overlap-aux-weight` / `--aux-weight` | 0 / 0 | 0 (pure `ℓ2`) | D4; suggested arm-B start `0.02 / 0.1` |
| `--num-samples` | none (full 1M) | 600k, <50% used | few-shot ablation knob |

Everything else (`--augment`, `--file-shuffle`, `--vram-fraction`, `--gpu-util-cap`, `--save-every`,
`--snapshot-every`, `--keep-recent`, auto-resume, SIGTERM/SIGHUP checkpointing) is
`direct_train_claude`'s, unchanged.

**Recommended first run** (after the L4 frees up):

```bash
cd /nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3
PYTHONPATH="$PWD/FloorSet/iccad2026contest:$PWD/FloorSet:$PWD/partner" \
uv run python src/solver/flow_distill_claude.py \
  --teacher-checkpoint checkpoints/flow_matching_v3/final.pt \
  --data-path FloorSet --checkpoint-dir checkpoints/flow_distill_v1_st1 \
  --steps-target 1 --max-steps 8000 --batch-size 12 --amp \
  --vram-fraction 0.5 --gpu-util-cap 0.9 --probe-every 250
```

## 5. Training budget

Per step: 3 no-grad forwards (teacher@t1, fast@t1, slow@t2) + 1 forward + 1 backward ≈ 6 forward-
equivalents, vs ≈ 3.5 for `flow_train_claude`. The v3 run does 8.3 it/s at batch 12 on the L4, so
expect **≈ 4–5 it/s** on a free card, ≈ 2.5 it/s while sharing.

- paper-scale convergence (1–2k steps) → **4–8 minutes**
- the recommended 8k steps → **≈ 30 min**
- the 20k-step ceiling → **≈ 1.5 h**

Comfortably inside the "GPU-hour to half-day" envelope, and cheap enough that arm A / arm B /
few-shot can all be run rather than argued about. The binding constraint is **evaluator** time
(full-100 runs), not training time — which is another reason Gate 0 comes first.

## 6. Acceptance gates

Terminal metric is the full-100 evaluator `no_runtime` score, never raw loss. Control for every
comparison: the promoted D+F 1M st8 pipeline, **no_runtime 1.1443 / proj 0.8932**, via
`scripts/probes/run_flow_variant_eval.sh` (which pins `PARTNER_FLOW_SLOTS=10 PARTNER_FLOW_STEPS=8
PARTNER_FLOW_SOLVER=euler`, `PARTNER_BUDGET_MAX=3.5 PARTNER_DDIM_STEPS=25 PARTNER_DIRECT_MIN=2.0`).

**Gate 0 — free baseline (no training). Run this first.**
Same harness, `PARTNER_FLOW_STEPS=2` then `=1`, teacher unchanged.
- If st2 lands within noise of 1.1443 → **the distillation is unnecessary**; bank the NFE cut and go
  straight to Gate 3. This is the outcome M1 makes plausible and it must be excluded before spending
  anything.
- If st2 regresses materially → the regression size is the exact budget the student must recover,
  and Gate 2's kill threshold is calibrated from it.

**Gate 1 — unit.** `tests/test_partner_flow_distill.py`, **22 tests, all passing**: window ordering
and `[0,1]` containment across jitter × shift × ladder, ladder/max-skip derivation and rejection of
unusable grids, empirical and degenerate `teacher_frac`, shift monotone / endpoint-fixing /
noise-concentrating / identity at `s=1`, straight field is an exact fixed point, the Eq. 11
composition identity, target degeneration to each endpoint, and tag acceptance by
`flow_train_claude.checkpoint_method`.

**Gate 2 — offline fidelity (minutes, no evaluator).** Student vs teacher at matched steps on the
official validation inputs, via the existing
`scripts/probes/flow_candidate_probe.py --flow-checkpoint <student> --flow-steps 1 2 4 8`,
reading `raw_overlap` / `raw_hpwl_proxy` / `raw_*_violations` / `anchor_exact`.
**Kill rule:** the student at `steps_target` must beat the *teacher at the same step count* on
`raw_overlap` by a clear margin. If it does not, the distillation did nothing and Gate 3 is a waste
of evaluator time. **Do not promote on Gate 2** — it is a filter, not evidence.

**Gate 3 — terminal, full-100.** `FLOW_CKPT=<student> PARTNER_FLOW_STEPS=<target>`, everything else
at the promoted config. Promote only on `no_runtime ≤ 1.1443` (within run-to-run noise) **and**
feasibility 100/100. Report st1 / st2 / st4 against teacher st8 in one table.

**Gate 4 — budget-layer conversion (the actual payoff).** A no-runtime tie at fewer NFE is not yet
a win; the freed time must be *converted* and the conversion must clear the budget layer (score
evidence + reusable risk signals + raw runtime tail, `block_count ≥ 100`):
1. st_target, slots unchanged → pure runtime win, `no_runtime` must not move;
2. st_target with `PARTNER_FLOW_SLOTS` raised to matched wall-clock → more candidates;
3. st_target with the freed time returned to `PARTNER_BUDGET_MAX` → more SA.

**Prerequisite measurement, currently missing.** Nobody has measured what fraction of the ~3.5 s
per-case budget the flow channel actually consumes. If `_sample_flow_preds` is (say) 0.3 s of 3.5 s,
then even a perfect 8× NFE cut frees ~0.26 s/case and Gate 4 is marginal — which would make this
whole line low-priority regardless of how well Gates 0–3 go. **Measure it before committing GPU
time**: time `my_opt_claude._sample_flow_preds` over a handful of cases at st8 vs st1. This is the
single number that decides the project's value, and it is a ten-minute job.

## 7. What was verified today

- `src/solver/flow_distill_claude.py`, 646 lines, syntax-clean, no unused imports.
- **`uv run pytest tests/test_partner_flow_distill.py` → 22 passed**, including: empirical
  `teacher_frac` 0.405 vs 0.4 nominal; **straight field is an exact fixed point of `scfm_target`**;
  **`z_t1 + (d1+d2)·target ==` two Euler sub-steps** (Eq. 11).
- **No regression**: the full flow suite (`test_partner_flow_matching` +
  `_flow_training` + `_flow_probe` + `_flow_distill`) is **86 passed** after adding
  `flow_distill_v1` to `FLOW_METHODS`.
- **CPU dry run, real v1 teacher, real data**: 20 steps in 30 s at batch 2 (1.4 it/s), 107.5M
  params, loss finite and non-degenerate, `drift` (student vs `V_near`) rising from 0 as expected.
- **Resume path**: auto-resume from `latest.pt` restored step 20 and continued to 24, with a
  different `--steps-target` (ladder rebuilt correctly).
- **Option matrix, 2–3 steps each, all clean**: `--teacher-frac 0.0` / `1.0` (degenerate slices skip
  the unused forward; skip mean 1.00 for the all-teacher arm), `--aux-weight`,
  `--overlap-aux-weight`, `--aux-time-weight t2`, `--student-self-cond 1`, `--shift 2.5–4.5`,
  `--impose-known 0`, `--grid-jitter 0 --grid-steps 16`.
- **Checkpoint compatibility**: the saved payload carries
  `{model, ema, ema_fast, optimizer, sched, step, model_config, args}` with
  `training_method="flow_distill_v1"`; `flow_train_claude.checkpoint_method` accepts it (tag added to
  `FLOW_METHODS`), `DirectModelConfig`/`DirectDenoiser` rebuild from `model_config`, the production
  line `ckpt.get("ema") or ckpt["model"]` loads, the probe's `EMA.copy_to` line loads, and
  `sample_flow` produces finite `z` at steps 1/2/4/8 with `nfe` = steps.
- **M1 step curve** (`scripts/probes/flow_step_curve_probe.py`), the table in §0.

**Not verified:** no GPU run (the L4 is held by the v3 job at `--vram-fraction 0.85` until ~14:00);
no convergence evidence; no evaluator evidence. The M1 numbers are on samples the v1 teacher saw in
training (index 500k of the 1M set), so the *absolute* values are optimistic — the within-model
step-count *ratios*, which are what the argument rests on, are not affected by that.

## 8. Risks

1. **Emergent 1-step is the load-bearing assumption.** SCFM never trains the full-path jump (D3
   partially fixes this). If the field does not straighten enough, st1 stays at 8× overlap. Mitigated
   by `--steps-target 2` as the safe arm — st2 is only 3.2× and the paper's own headline students are
   3–8 step.
2. **Raw-metric wins may not transfer** (the repo's standing law). Gate 2 is explicitly a filter,
   not evidence; only Gate 3 promotes.
3. **The payoff may be too small to matter** — see the §6 prerequisite measurement. This is the
   biggest risk and the cheapest to retire.
4. **Self-conditioning mismatch at ≥2 steps** (A2): the student's LHS is trained without self-cond
   but the 2-step sampler's second call supplies one. `--student-self-cond 1` is the ablation.
5. **v3 is not the teacher yet.** The script takes any checkpoint; the teacher must be whichever of
   v1/v3 wins its own full-100 gate. Distilling a loser wastes the run.
6. **Distribution shift** — the paper's few-shot result is over *prompts*; our conditioning varies in
   graph structure and block count (21–120), a much larger space. Expect few-shot to work worse here.
   `--num-samples` makes it a cheap ablation rather than an assumption; the default is the full set,
   which costs nothing extra at these step counts.

## 9. Fallbacks

The paper's method needs no Flux-specific structure — no step-size embedding, no CFG embedding, no
LoRA requirement, no architecture change — so **no fallback is needed for feasibility**. These are
fallbacks for *failure to converge*:

- **F1 — progressive distillation** (Salimans & Ho; the CO-diffusion 16× result). Reachable from the
  same script with `--teacher-frac 1.0 --max-skip 2`, run in stages, re-pointing
  `--teacher-checkpoint` at the previous stage's output each time. Remark 1 of the paper notes
  `k = 0` recovers a progressive-style scheme; `k = 1` with the smallest ladder is the teacher-only
  variant. No new code.
- **F2 — shortcut fine-tune** (add a real `d` embedding, Frans et al.). Strictly stronger in
  principle, but changes `DirectModelConfig` and therefore every loader listed in §2. Last resort
  only, and not before the deadline.
