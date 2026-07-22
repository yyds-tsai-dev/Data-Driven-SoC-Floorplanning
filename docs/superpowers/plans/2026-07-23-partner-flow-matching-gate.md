# Partner Flow Matching Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a conditional Flow Matching candidate generator against the current Direct-v2 50-NFE DDIM baseline, then integrate it only if low-NFE candidates improve quality, runtime, or complementarity under the same partner deadline.

**Architecture:** Reuse `DirectDenoiser`, its graph conditioning, layout representation, hard-anchor channels, D4 augmentation, and auxiliary geometry losses. Replace only the diffusion path/target with a straight conditional flow and add Euler/Heun samplers that report exact neural-function evaluations. Candidate-only evaluation precedes any partner integration.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, existing Direct-v2 trainer and partner evaluator.

## Global Constraints

- Complete `2026-07-23-partner-candidate-source-foundation.md` first.
- Direct-v2 50-NFE DDIM remains the control; do not replace it by default.
- Use training data only for fitting and overfit probes; evaluation labels are diagnostic only.
- Compare both fixed accumulated training samples and fixed training wall-clock.
- Report exact NFE and measured sampling latency; Heun step count alone is not a speed claim.
- Flow candidates replace Direct/refine capacity under the same case deadline, GPU batch cap, prescreen count, workers, and refine slots.
- Do not combine Flow Matching with retrieval until both independent gates pass.

---

## File Structure

- Create `partner/flow_matching_claude.py`: straight-path objective helpers and Euler/Heun samplers.
- Create `partner/flow_train_claude.py`: Direct-v2-compatible trainer override and tagged checkpoints.
- Create `scripts/probes/flow_candidate_probe.py`: candidate-only NFE/quality/latency matrix.
- Modify `partner/my_opt_claude.py`: opt-in flow checkpoint and fixed-quota candidate source.
- Create `scripts/probes/run_flow_matching_gate.sh`: smoke, candidate-only, and end-to-end gate runner.
- Create `tests/test_partner_flow_matching.py`, `tests/test_partner_flow_training.py`, and `tests/test_partner_flow_integration.py`.

### Task 1: Straight-Path Objective and Samplers

**Files:**
- Create: `partner/flow_matching_claude.py`
- Test: `tests/test_partner_flow_matching.py`

**Interfaces:**
- Produces: `flow_path(z0, noise, t) -> (z_t, velocity_target)`.
- Produces: `endpoint_from_velocity(z_t, velocity, t) -> z0_hat`.
- Produces: `sample_flow(model, condition, steps, solver, generator, z_known, known_mask) -> FlowSample`.
- `FlowSample` contains `z: torch.Tensor`, `nfe: int`, and `solver: str`.

- [ ] **Step 1: Write failing path, Euler, Heun, mask, and anchor tests**

```python
from types import SimpleNamespace

import torch

from flow_matching_claude import endpoint_from_velocity, flow_path, sample_flow


class ConstantVelocity(torch.nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.register_buffer("velocity", velocity)
        self.config = SimpleNamespace(z_dim=velocity.shape[-1], timesteps=1000)

    def forward(self, z, t, node_feat, adj, mask, rel_feat=None, self_cond=None):
        return self.velocity.expand_as(z) * mask.unsqueeze(-1)


def _condition(batch=1, blocks=2):
    return {
        "node_feat": torch.zeros(batch, blocks, 1),
        "adj": torch.eye(blocks).expand(batch, -1, -1),
        "mask": torch.tensor([[True, False]]).expand(batch, -1),
        "rel_feat": torch.zeros(batch, blocks, blocks, 0),
    }


def test_flow_path_endpoint_reconstruction_is_exact():
    z0 = torch.tensor([[[2.0, -1.0]]])
    noise = torch.tensor([[[0.5, 3.0]]])
    t = torch.tensor([0.25])
    z_t, target = flow_path(z0, noise, t)
    torch.testing.assert_close(endpoint_from_velocity(z_t, target, t), z0)


def test_euler_reports_one_nfe_per_step_and_masks_padding():
    model = ConstantVelocity(torch.ones(1, 2, 4))
    sample = sample_flow(model, _condition(), steps=4, solver="euler",
                         generator=torch.Generator().manual_seed(1))
    assert sample.nfe == 4
    assert torch.equal(sample.z[:, 1], torch.zeros_like(sample.z[:, 1]))


def test_heun_reports_two_nfe_per_step():
    model = ConstantVelocity(torch.ones(1, 2, 4))
    sample = sample_flow(model, _condition(), steps=4, solver="heun",
                         generator=torch.Generator().manual_seed(1))
    assert sample.nfe == 8


def test_known_channels_are_exact_at_final_time():
    model = ConstantVelocity(torch.zeros(1, 2, 4))
    known = torch.tensor([[[7.0, 5.0, 1.0, 0.0], [0.0] * 4]])
    mask = torch.tensor([[[True, True, True, False], [False] * 4]])
    sample = sample_flow(model, _condition(), steps=4, solver="euler",
                         generator=torch.Generator().manual_seed(1),
                         z_known=known, known_mask=mask)
    torch.testing.assert_close(sample.z[mask], known[mask])
```

- [ ] **Step 2: Verify missing-module failure**

Run: `uv run pytest tests/test_partner_flow_matching.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the objective helpers**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class FlowSample:
    z: torch.Tensor
    nfe: int
    solver: str


def _broadcast_time(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], *([1] * (z.ndim - 1))).to(z)


def flow_path(z0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
    tau = _broadcast_time(t, z0)
    return (1.0 - tau) * noise + tau * z0, z0 - noise


def endpoint_from_velocity(z_t: torch.Tensor, velocity: torch.Tensor,
                           t: torch.Tensor) -> torch.Tensor:
    tau = _broadcast_time(t, z_t)
    return z_t + (1.0 - tau) * velocity
```

- [ ] **Step 4: Implement Euler/Heun with a single known-channel path**

`sample_flow()` must draw `known_noise` once, not once per step. At every `t_next`, impose `(1-t_next)*known_noise + t_next*z_known`; at `t=1`, hard channels are exactly known.

```python
@torch.no_grad()
def sample_flow(model, condition: Mapping[str, torch.Tensor], steps: int,
                solver: str, generator=None, z_known=None, known_mask=None):
    if steps <= 0 or solver not in {"euler", "heun"}:
        raise ValueError("steps must be positive and solver must be euler or heun")
    mask = condition["mask"]
    batch, blocks = mask.shape
    z = torch.randn((batch, blocks, model.config.z_dim), device=mask.device,
                    generator=generator) * mask.unsqueeze(-1)
    known_noise = torch.randn(z.shape, device=z.device, generator=generator) \
        if z_known is not None else None
    nfe = 0
    self_condition = None

    def velocity(state, scalar_t, self_cond=None):
        nonlocal nfe
        t_model = torch.full((batch,), scalar_t * (model.config.timesteps - 1),
                             device=z.device)
        nfe += 1
        return model(state, t_model, condition["node_feat"], condition["adj"], mask,
                     rel_feat=condition.get("rel_feat"), self_cond=self_cond)

    for step in range(steps):
        t0 = step / steps
        t1 = (step + 1) / steps
        dt = t1 - t0
        v0 = velocity(z, t0, self_condition)
        self_condition = endpoint_from_velocity(
            z, v0, torch.full((batch,), t0, device=z.device)
        ).detach()
        if z_known is not None and known_mask is not None:
            self_condition = torch.where(known_mask, z_known, self_condition)
        if solver == "euler":
            z_next = z + dt * v0
        else:
            predicted = z + dt * v0
            v1 = velocity(predicted, t1, self_condition)
            z_next = z + 0.5 * dt * (v0 + v1)
        if z_known is not None and known_mask is not None:
            imposed = (1.0 - t1) * known_noise + t1 * z_known
            z_next = torch.where(known_mask, imposed, z_next)
        z = z_next * mask.unsqueeze(-1)
    return FlowSample(z=z, nfe=nfe, solver=solver)
```

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/test_partner_flow_matching.py -q`

Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add partner/flow_matching_claude.py tests/test_partner_flow_matching.py
git commit -m "feat: add conditional flow matching sampler"
```

### Task 2: Direct-v2-Compatible Flow Training

**Files:**
- Create: `partner/flow_train_claude.py`
- Test: `tests/test_partner_flow_training.py`

**Interfaces:**
- Produces: `flow_train_step(model, ema, unused_schedule, batch, args, rng, amp_dtype, amp_on)` matching the existing trainer callback.
- Checkpoints retain `model_config`, `model`, `ema`, optimizer state, and set `args.training_method == "flow_matching_v1"`.
- Uses `z_repr="xyaspect"` and the same Direct-v2 condition/augmentation/geometry helpers.

- [ ] **Step 1: Write failing deterministic loss and checkpoint-tag tests**

```python
from types import SimpleNamespace

import torch

from flow_matching_claude import endpoint_from_velocity, flow_path
from flow_train_claude import checkpoint_method, masked_flow_loss


def test_zero_error_flow_velocity_has_zero_primary_loss():
    z0 = torch.tensor([[[1.0, 2.0, 0.0, 0.0]]])
    noise = torch.zeros_like(z0)
    t = torch.tensor([0.5])
    z_t, target = flow_path(z0, noise, t)
    mask = torch.tensor([[True]])
    loss, endpoint = masked_flow_loss(target, target, z_t, z0, t, mask)
    assert loss.item() == 0.0
    torch.testing.assert_close(endpoint, z0)


def test_checkpoint_method_rejects_diffusion_checkpoint():
    try:
        checkpoint_method({"args": {"training_method": "diffusion"}})
    except ValueError as exc:
        assert "flow_matching_v1" in str(exc)
    else:
        raise AssertionError("diffusion checkpoint accepted as flow")
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_partner_flow_training.py -q`

Expected: FAIL importing `flow_train_claude`.

- [ ] **Step 3: Implement primary loss and checkpoint guard**

```python
def checkpoint_method(checkpoint):
    method = checkpoint.get("args", {}).get("training_method")
    if method != "flow_matching_v1":
        raise ValueError("checkpoint must declare training_method=flow_matching_v1")
    return method


def masked_flow_loss(predicted, target, z_t, z0, t, mask):
    weight = mask.unsqueeze(-1).to(predicted.dtype)
    velocity_loss = ((predicted - target).square() * weight).sum() \
        / (weight.sum().clamp_min(1) * predicted.shape[-1])
    endpoint = endpoint_from_velocity(z_t, predicted, t)
    return velocity_loss, endpoint
```

- [ ] **Step 4: Implement `flow_train_step` by changing only the path target**

Reuse from `direct_train_claude`: `augment_batch`, `fast_condition`, `known_target_positions_from_fp`, `known_z_channels`, `overlap_fraction`, `boundary_touch`, `cluster_gap`, `mib_aspect`, and `z_to_rectangles`. Reuse `hpwl_pair` and the large-case weight formula from `direct_train_v2_claude`.

The changed core must be:

```python
t = torch.rand((batch_size,), device=device, generator=rng)
noise = torch.randn(z0.shape, device=device, generator=rng)
z_t, velocity_target = flow_path(z0, noise, t)
z_t = z_t * mask.unsqueeze(-1)
t_model = t * (model.config.timesteps - 1)
velocity = model(z_t, t_model, cond["node_feat"], cond["adj"], mask,
                 rel_feat=cond["rel_feat"], self_cond=self_condition)
velocity_loss, z0_hat = masked_flow_loss(
    velocity.float(), velocity_target, z_t, z0, t, mask
)
```

For self-conditioning, use a no-grad first velocity prediction and `endpoint_from_velocity(z_t, first_velocity, t)`; project known channels into that endpoint. Gate geometry losses by `t**2`, because endpoint estimates become more reliable closer to data time. Keep Direct-v2 HPWL, overlap, boundary, cluster, MIB, boundary-node weighting, EMA, checkpoint pruning, signal handling, and shared-GPU throttling unchanged.

Set before invoking the reused V1 training loop:

```python
import direct_train_claude as V1

original_parse = V1.parse_args


def parse_flow_args():
    args = original_parse()
    args.training_method = "flow_matching_v1"
    if not hasattr(args, "hpwl_loss_weight"):
        args.hpwl_loss_weight = 0.30
    return args


V1.parse_args = parse_flow_args
V1.train_step = flow_train_step
V1.main()
```

- [ ] **Step 5: Add a one-step CPU smoke test**

Append this minimal differentiability and checkpoint smoke:

```python
def test_flow_primary_loss_backpropagates_and_checkpoint_is_tagged():
    predicted = torch.nn.Parameter(torch.zeros(1, 2, 4))
    z0 = torch.ones(1, 2, 4)
    noise = torch.zeros_like(z0)
    t = torch.tensor([0.5])
    z_t, target = flow_path(z0, noise, t)
    loss, _endpoint = masked_flow_loss(
        predicted, target, z_t, z0, t, torch.tensor([[True, False]])
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert checkpoint_method({"args": {"training_method": "flow_matching_v1"}}) \
        == "flow_matching_v1"
```

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest tests/test_partner_flow_training.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add partner/flow_train_claude.py tests/test_partner_flow_training.py
git commit -m "feat: train direct architecture with flow matching"
```

### Task 3: Candidate-Only NFE and Quality Probe

**Files:**
- Create: `scripts/probes/flow_candidate_probe.py`
- Test: `tests/test_partner_flow_probe.py`

**Interfaces:**
- Flags: `--direct-checkpoint`, `--flow-checkpoint`, `--cases`, `--samples`, `--flow-steps`, `--solvers`, `--output`.
- Output JSON: one row per case/model/solver/NFE with latency, peak memory, overlap, HPWL proxy, boundary/group/MIB proxy, and best-of-K.

- [ ] **Step 1: Write parser and matrix tests**

```python
from scripts.probes.flow_candidate_probe import build_matrix


def test_probe_matrix_reports_nfe_not_only_steps():
    rows = build_matrix(flow_steps=[4, 8], solvers=["euler", "heun"])
    assert ("flow", "euler", 4, 4) in rows
    assert ("flow", "heun", 4, 8) in rows
    assert ("direct", "ddim", 50, 50) in rows
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_partner_flow_probe.py -q`

Expected: FAIL importing the missing script.

- [ ] **Step 3: Implement the fixed matrix and synchronized timing**

```python
def build_matrix(flow_steps, solvers):
    rows = [("direct", "ddim", 50, 50)]
    for solver in solvers:
        for steps in flow_steps:
            rows.append(("flow", solver, steps, steps if solver == "euler" else 2 * steps))
    return rows
```

Use `torch.cuda.synchronize()` immediately before and after timed CUDA sampling. Warm each model once outside measurement. Use identical case list, K, seeds, conditioning, and rectangle decoder for every row. Record cold load separately from warm sampling.

- [ ] **Step 4: Run tests and a small probe**

Run: `uv run pytest tests/test_partner_flow_probe.py -q`

Expected: PASS.

Run:

```bash
uv run python scripts/probes/flow_candidate_probe.py \
  --direct-checkpoint partner/checkpoints/direct_v2/latest.pt \
  --flow-checkpoint checkpoints/flow_matching_probe/latest.pt \
  --cases 21 68 115 --samples 8 \
  --flow-steps 4 8 16 25 50 --solvers euler heun \
  --output artifacts/flow_matching/candidate_probe.json
```

Expected: JSON contains the Direct DDIM-50 row and every Flow solver/NFE row for all three cases.

- [ ] **Step 5: Commit**

```bash
git add scripts/probes/flow_candidate_probe.py tests/test_partner_flow_probe.py
git commit -m "feat: compare flow and DDIM candidate efficiency"
```

### Task 4: Opt-In Partner Flow Candidate Source

**Files:**
- Modify: `partner/my_opt_claude.py`
- Test: `tests/test_partner_flow_integration.py`

**Interfaces:**
- New env: `FLOW_CKPT`, absent by default.
- New env: `PARTNER_FLOW_SLOTS`, default `0`.
- New env: `PARTNER_FLOW_STEPS`, default `8`.
- New env: `PARTNER_FLOW_SOLVER`, default `euler`.
- Produces: `_sample_flow_preds(self, n, at, cons, tpos, b2b, p2b, pins, K) -> CandidateBatch`.

- [ ] **Step 1: Write disabled-default, checkpoint-tag, and quota tests**

```python
from candidate_supply_claude import allocate_quotas
from flow_train_claude import checkpoint_method


def test_flow_disabled_keeps_direct_capacity():
    assert allocate_quotas(12, {"direct": 12, "flow": 0}, ("direct",)) == {"direct": 12}


def test_flow_replaces_not_adds_capacity():
    got = allocate_quotas(12, {"direct": 6, "flow": 6}, ("direct", "flow"))
    assert got == {"direct": 6, "flow": 6}


def test_flow_loader_rejects_untagged_checkpoint():
    try:
        checkpoint_method({"args": {}})
    except ValueError:
        pass
    else:
        raise AssertionError("untagged checkpoint accepted")
```

- [ ] **Step 2: Run tests before integration**

Run: `uv run pytest tests/test_partner_flow_integration.py -q`

Expected: quota and guard tests PASS.

- [ ] **Step 3: Add an opt-in loader**

Initialize `self.flow_model = None`. Load only when `FLOW_CKPT` exists and `PARTNER_FLOW_SLOTS > 0`; call `checkpoint_method()` before loading weights. A failed flow load disables only Flow Matching and leaves Direct-v2 unchanged.

- [ ] **Step 4: Sample through the shared candidate boundary**

Reuse `fast_condition`, `known_z_channels`, and `z_to_rectangles`. Call:

```python
sample = sample_flow(
    self.flow_model, condition, steps=_env_int("PARTNER_FLOW_STEPS", 8),
    solver=os.environ.get("PARTNER_FLOW_SOLVER", "euler"), generator=generator,
    z_known=z_known.expand(flow_batch, -1, -1),
    known_mask=known.expand(flow_batch, -1, -1),
)
```

Convert to NumPy rectangles, wrap in `CandidateBatch(source="flow", predictions=predictions, generation_s=elapsed, metadata={"nfe": sample.nfe})`, merge with Direct candidates, rank once with `rank_predictions()`, and truncate to the unchanged refine capacity.

- [ ] **Step 5: Verify focused and full tests**

Run: `uv run pytest tests/test_partner_flow_matching.py tests/test_partner_flow_training.py tests/test_partner_flow_integration.py -q`

Expected: all focused tests PASS.

Run: `uv run pytest`

Expected: full suite PASS.

Run: `bash scripts/validate.sh`

Expected: interface validation succeeds with Flow disabled by default.

- [ ] **Step 6: Commit**

```bash
git add partner/my_opt_claude.py tests/test_partner_flow_integration.py
git commit -m "feat: add time-neutral flow candidates"
```

### Task 5: Training and End-to-End Gate Runner

**Files:**
- Create: `scripts/probes/run_flow_matching_gate.sh`
- Create after results: `docs/experiments/2026-07-23-flow-matching-gate.md`

**Interfaces:**
- Stages: CPU smoke, small training-only overfit, fixed-sample training, fixed-wall-clock training, candidate-only matrix, partner D/F/D+F full-100 evaluation.

- [ ] **Step 1: Write the staged runner with explicit checkpoints**

The runner must stop at each failed gate and print the exact next command rather than launching a week-long job automatically:

```bash
uv run pytest tests/test_partner_flow_matching.py \
  tests/test_partner_flow_training.py tests/test_partner_flow_probe.py \
  tests/test_partner_flow_integration.py -q

uv run python partner/flow_train_claude.py \
  --data-path FloorSet --checkpoint-dir checkpoints/flow_matching_overfit \
  --num-samples 256 --batch-size 16 --max-steps 2000 \
  --d-model 128 --layers 2 --heads 4 --device cuda --amp

uv run python scripts/probes/flow_candidate_probe.py \
  --direct-checkpoint partner/checkpoints/direct_v2/latest.pt \
  --flow-checkpoint checkpoints/flow_matching_overfit/latest.pt \
  --cases 21 68 115 --samples 8 --flow-steps 4 8 16 25 50 \
  --solvers euler heun --output artifacts/flow_matching/overfit_probe.json
```

- [ ] **Step 2: Run the small overfit gate**

Expected: training loss and position error decrease; 8/16-NFE samples are finite, anchors exact, and repaired candidate metrics improve over initial random layouts. Failure stops long training.

- [ ] **Step 3: Run two fair long-training comparisons**

Run one Flow job matched to Direct-v2 by 800,000 updates:

```bash
uv run python partner/flow_train_claude.py \
  --data-path FloorSet --checkpoint-dir checkpoints/flow_matching_v1 \
  --batch-size 12 --max-steps 800000 --d-model 640 --layers 14 \
  --heads 10 --node-feat-dim 32 --lr 8e-5 --warmup 4000 \
  --ema-decay 0.9998 --device cuda --amp
```

For the fixed-wall-clock comparison, launch fresh Direct-v2 and Flow runs with the same GPU duty cap, batch size, and 24-hour external scheduler allocation; terminate both with SIGTERM so their interruption-safe checkpoints are written:

```bash
uv run python partner/direct_train_v2_claude.py \
  --data-path FloorSet --checkpoint-dir checkpoints/direct_v2_24h --fresh \
  --batch-size 12 --max-steps 800000 --gpu-util-cap 0.75 --amp

uv run python partner/flow_train_claude.py \
  --data-path FloorSet --checkpoint-dir checkpoints/flow_matching_24h --fresh \
  --batch-size 12 --max-steps 800000 --gpu-util-cap 0.75 --amp
```

The job scheduler, not a blocking shell sleep, enforces the 24-hour limit. Record parameter count, processed examples, updates, GPU model, elapsed time, and checkpoint hash.

- [ ] **Step 4: Run candidate-only F3 matrix**

Proceed to partner integration only if `<=16` NFE Flow matches DDIM-50 quality, same-NFE Flow improves quality, or Flow has a stable complementary winner set after cheap repair.

- [ ] **Step 5: Run fixed-budget D, F, and D+F full-100 evaluations**

Use the same candidate/refine total for all configurations:

```bash
PARTNER_FLOW_SLOTS=0 bash scripts/partner_eval_cont.sh flow_D_control

FLOW_CKPT=checkpoints/flow_matching_v1/latest.pt \
PARTNER_FLOW_SLOTS=15 PARTNER_FLOW_STEPS=8 PARTNER_FLOW_SOLVER=euler \
DIRECT_OFF=1 bash scripts/partner_eval_cont.sh flow_F_only

FLOW_CKPT=checkpoints/flow_matching_v1/latest.pt \
PARTNER_FLOW_SLOTS=6 PARTNER_FLOW_STEPS=8 PARTNER_FLOW_SOLVER=euler \
bash scripts/partner_eval_cont.sh flow_DF_flow6
```

Confirm from metadata that D+F has the same total refine slots as D.

- [ ] **Step 6: Write the decision report and commit**

Report no-runtime, average/median/p90/max/sum runtime, exact sampling latency/NFE, alpha-median projection with hidden-mismatch warning, weighted per-case deltas, source winner contribution, and repeated-seed uncertainty.

```bash
git add scripts/probes/run_flow_matching_gate.sh docs/experiments/2026-07-23-flow-matching-gate.md
git commit -m "docs: record flow matching gate results"
```

## Gate F4 Decision

Promote Flow Matching only if paired full-100 evidence shows one of:

1. Better no-runtime score without worse runtime-aware projection.
2. Equivalent no-runtime score with materially lower sampling or total runtime.
3. A stable complementary winner set that improves D+F under unchanged capacity.

Otherwise retain Direct-v2 DDIM-50 and record Flow Matching as disproven for this architecture/training budget. Do not combine a failed Flow source with retrieval.

If Flow F4 and retrieval R4 both pass, write a separate D+R+F portfolio plan with fixed source quotas and feature-gated allocation as distinct ablations. Do not add retrieval conditioning to the first Flow implementation.
