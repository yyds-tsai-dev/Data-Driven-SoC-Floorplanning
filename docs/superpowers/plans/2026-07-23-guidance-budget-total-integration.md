# Physics-Guidance × Budget × Flow 總整合實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 beta(7/31 17:00)前把三個正交槓桿落地:(A) 尾段 runtime 預算重塑(投影 −0.23~−0.30)、(B) training-free physics-guided sampling(抬預測品質天花板、降 refine 成本)、(C) flow matching 整合準備(S-1 修復 + my_opt 接線),外加品質補償(POOL 46)、GPU 減步與 beta 打包。

**Architecture:** 全部改動 opt-in、預設關閉(bare defaults = 現役行為);promotion 一律走 evaluator evidence(full-100 `total_score_no_runtime` + runtime tail)。guidance 是純採樣端(不動任何訓練/checkpoint):在 `sample_direct` 的 DDIM 每步 x̂₀ 上做小步能量梯度下降再反推 eps;能量自含於新模組 `src/solver/physics_guidance_claude.py`(不 import v11 的 layout_losses,因其綁 `DiffusionGraphInputs`)。

**Tech Stack:** Python 3.12 / PyTorch / numpy / pytest(`uv run pytest`,repo root)。評測經 `FloorSet/iccad2026contest/../scripts/iccad2026_evaluate.py`(見 Task 1 腳本)。

## Global Constraints

- **不准加時間**:任何 treatment 的 per-case runtime 不得超過現役 budget 曲線;guidance 只能用 GPU 段(被 column restarts 遮蔽的窗口)。
- **不准 hardcode test_id**:自適應只能用可重用 instance 統計(block_count、boundary-coded 比例、overlap 密度)。
- **golden 標籤(fp_sol)絕不進推理/選擇**;hard anchors 只用 solve() 可見的 preplaced xywh / fixed wh。
- 所有新 env 旗標預設 off;bare defaults 必須 bit-for-bit 重現現役行為。
- 訓練中的 `flow_matching_v1`(worktree,~7/25 完)**不得中止、不得改其訓練端代碼**(`flow_train_claude.py` 凍結);只可改採樣端 `flow_matching_claude.py`(運行中進程不受磁碟改動影響)。
- direct_v2 checkpoint 一律用既有 snapshot `partner/checkpoints/direct_v2_cont/eval_retrieval_direct_control.pt`(step 1,139,000),與 R4 control 可比。
- Baseline 錨點:`artifacts/partner_eval/cont_retrieval_direct_control.json` = no-runtime **1.1258**,budget MAX=24。
- 計分 env(所有 full-100 對照共用):`VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3`。

## 依賴圖(無天數;可並行者並行,能立即做的立即做)

```
[T1 budget 掃描發跑]──(背景 ~1h)──▶[T8 定案 runs]──▶[T9 guidance full-100 gate]──▶[T13 組合定案+打包]
[T2 S-1 修復]────────(立即 5min)                                    ▲
[T3 能量核]──▶[T4 guide_x0]──▶[T5 sample_direct 注入]──▶[T6 my_opt 接線]──▶[T7 快 probe]
[T10 flow my_opt 整合]───(立即可做,overfit ckpt 驗證;正式 gate 等 v1 訓完)
[T11 guided-FM 注入]────(T4 完成後可做;probe 等 v1)
[T12 numba profile]─────(獨立;profile 先行)
```

---

### Task 1: Budget 尾段掃描 — 立即發跑(純 env,不改代碼)

**Files:**
- Create: `scripts/probes/run_budget_scan.sh`
- Output: `artifacts/partner_eval/budget_scan_max{8,6,4}.json`

**Interfaces:**
- Produces: 三份 full-100 JSON(`total_score_no_runtime` + per-case runtime),供 Task 8 判讀 Q-time 彈性與 direct cliff。

設計:只動 `PARTNER_BUDGET_MAX`(24→8/6/4),其餘曲線參數不動 — MAX=8 只 cap n≳100、MAX=6 cap n≳94、MAX=4 cap n≳84,歸因乾淨。基線(MAX=24)= 既有 control JSON,不重跑。

- [ ] **Step 1: 寫掃描腳本**

```bash
#!/bin/bash
# Budget tail scan: PARTNER_BUDGET_MAX in {8,6,4}, all else = scoring env.
# Baseline (MAX=24) = artifacts/partner_eval/cont_retrieval_direct_control.json
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/FloorSet/iccad2026contest:$ROOT/FloorSet"
export DIRECT_CKPT="$ROOT/partner/checkpoints/direct_v2_cont/eval_retrieval_direct_control.pt"
export VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 \
       PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
cd "$ROOT/FloorSet/iccad2026contest"
for MAX in 8 6 4; do
  echo "=== PARTNER_BUDGET_MAX=$MAX ==="
  PARTNER_BUDGET_MAX=$MAX uv run python "$ROOT/scripts/iccad2026_evaluate.py" \
    --data-path ../ \
    --evaluate "$ROOT/src/solver/my_opt_claude.py" \
    --output "$ROOT/artifacts/partner_eval/budget_scan_max${MAX}.json" \
    2>&1 | tail -3
done
```

- [ ] **Step 2: 發跑(背景,~45-75 分鐘)**

Run: `bash scripts/probes/run_budget_scan.sh`(run_in_background)
Expected: 三份 JSON 落地;每份 100/100 feasible。

- [ ] **Step 3: Commit(腳本)**

```bash
git add scripts/probes/run_budget_scan.sh
git commit -m "feat: add budget tail scan runner"
```

---

### Task 2: Flow S-1 修復 — worktree 一行(立即)

**Files:**
- Modify: `/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/src/solver/flow_matching_claude.py:115`(`self_condition = endpoint_from_velocity(...).detach()` 之後、`if has_known:` 之前)
- Test: 同 worktree `tests/test_partner_flow_matching.py`

**Interfaces:**
- Produces: 採樣端 self-conditioning 的 aspect 通道 clamp(-3,3),與訓練端 `flow_train_claude.py:93` 對齊。訓練進程不受影響(運行中模組已載入記憶體;且訓練不呼叫 `sample_flow`)。

- [ ] **Step 1: 寫 failing test(worktree 的 tests/test_partner_flow_matching.py 追加)**

```python
def test_self_condition_aspect_channel_is_clamped():
    """Sampler must clamp the aspect channel of self-conditioning to match
    the training-side clamp (flow_train_claude.py), else train/sample skew."""
    captured = []

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(z_dim=4, timesteps=1000)

        def forward(self, z, t, node_feat, adj, mask, rel_feat=None, self_cond=None):
            if self_cond is not None:
                captured.append(self_cond.detach().clone())
            # huge velocity so the endpoint estimate's aspect channel exceeds 3
            v = torch.zeros_like(z)
            v[..., 2] = 100.0
            return v * mask.unsqueeze(-1)

    model = RecordingModel()
    sample_flow(model, _condition(), steps=3, solver="euler",
                generator=torch.Generator().manual_seed(0))
    assert captured, "self_cond was never passed back to the model"
    for sc in captured:
        assert sc[..., 2].abs().max() <= 3.0 + 1e-6
```

- [ ] **Step 2: 跑測試確認 FAIL**

Run: `cd /nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3 && uv run pytest tests/test_partner_flow_matching.py -q`
Expected: 新測試 FAIL(aspect 未 clamp,abs max ≈ 33+)。

- [ ] **Step 3: 一行修復**

在 `flow_matching_claude.py` 的 `self_condition = endpoint_from_velocity(...).detach()`(L111-115)之後插入:

```python
        self_condition[..., 2] = self_condition[..., 2].clamp(-3.0, 3.0)
```

- [ ] **Step 4: 跑測試確認全 PASS**

Run: `uv run pytest tests/test_partner_flow_matching.py tests/test_partner_flow_training.py -q`
Expected: 全部 PASS(原 4 + 新 1 + training 測試)。

- [ ] **Step 5: Commit(worktree branch)**

```bash
git add src/solver/flow_matching_claude.py tests/test_partner_flow_matching.py
git commit -m "fix: clamp sampler self-cond aspect channel"
```

---

### Task 3: `src/solver/physics_guidance_claude.py` — 能量核(TDD)

**Files:**
- Create: `src/solver/physics_guidance_claude.py`
- Test: `tests/test_partner_physics_guidance.py`

**Interfaces:**
- Produces: `GuidanceContext`(dataclass:`area[B,N]`, `mask[B,N]bool`, `known_mask[B,N,4]bool`, `scale[B]`, `boundary_code[B,N]int`(bitmask 1=左 2=右 4=頂 8=底,自參照排布 bbox,語義同 `my_opt_claude._viol_est`)、`pair_i[P]long`, `pair_j[P]long`, `pair_w[P]float`(b2b 上三角非零)、`group_id[B,N]int`)。
- Produces: `rects_from_z(z, ctx) -> (x, y, w, h)` 各 `[B,N]`,對 z 可微;**必須與 `diffusion_data.z_to_rectangles` 的 xyaspect 分支一致**:x=z0·S、y=z1·S(左下角)、aspect=exp(clamp(z2,±3))、w=√(area·aspect)、h=√(area/aspect)。
- Produces: `guidance_energy(z, ctx, w_overlap, w_boundary, w_hpwl, w_group) -> scalar tensor`(per-batch sum;pad 塊零貢獻)。
- Consumes: 無(standalone;刻意不 import v11 `layout_losses`)。

- [ ] **Step 1: 寫 failing tests**

```python
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "partner"))

from physics_guidance_claude import (GuidanceContext, guidance_energy,
                                     rects_from_z)


def _ctx(n=2, boundary=(0, 0), groups=(0, 0), pairs=None, areas=(100.0, 100.0)):
    B = 1
    N = n
    area = torch.tensor([list(areas)], dtype=torch.float32)
    mask = torch.ones(B, N, dtype=torch.bool)
    known = torch.zeros(B, N, 4, dtype=torch.bool)
    scale = torch.sqrt(area.sum(dim=1)).clamp_min(1.0)
    p = pairs or []
    return GuidanceContext(
        area=area, mask=mask, known_mask=known, scale=scale,
        boundary_code=torch.tensor([list(boundary)], dtype=torch.long),
        group_id=torch.tensor([list(groups)], dtype=torch.long),
        pair_i=torch.tensor([a for a, _, _ in p], dtype=torch.long),
        pair_j=torch.tensor([b for _, b, _ in p], dtype=torch.long),
        pair_w=torch.tensor([w for _, _, w in p], dtype=torch.float32),
    )


def _z(xy_pairs, aspect=0.0):
    # xy in *normalized* units (z0 = x/S); aspect = log(w/h)
    z = torch.zeros(1, len(xy_pairs), 4)
    for i, (x, y) in enumerate(xy_pairs):
        z[0, i, 0] = x
        z[0, i, 1] = y
        z[0, i, 2] = aspect
    return z


def test_rects_match_z_to_rectangles_semantics():
    ctx = _ctx(areas=(100.0, 25.0))
    z = _z([(0.5, 0.25), (0.0, 0.0)], aspect=0.0)
    x, y, w, h = rects_from_z(z, ctx)
    s = float(ctx.scale[0])
    assert abs(float(x[0, 0]) - 0.5 * s) < 1e-5
    assert abs(float(w[0, 0]) - 10.0) < 1e-4   # sqrt(100*1)
    assert abs(float(h[0, 1]) - 5.0) < 1e-4    # sqrt(25/1)


def test_overlap_energy_pushes_blocks_apart():
    ctx = _ctx()
    # two 10x10 blocks at identical corner -> full overlap
    z = _z([(0.1, 0.1), (0.1, 0.1)]).requires_grad_(True)
    e = guidance_energy(z, ctx, 1.0, 0.0, 0.0, 0.0)
    assert e.item() > 0.0
    g, = torch.autograd.grad(e, z)
    # symmetric stack: gradients on x must be opposite in sign or zero-sum
    assert abs(float(g[0, 0, 0] + g[0, 1, 0])) < 1e-5
    # separated blocks -> zero energy
    z2 = _z([(0.0, 0.0), (0.9, 0.9)])
    e2 = guidance_energy(z2, ctx, 1.0, 0.0, 0.0, 0.0)
    assert e2.item() < 1e-8


def test_boundary_energy_pulls_coded_block_to_wall():
    # block 0 coded left-wall (bit 1), block 1 free and further left
    ctx = _ctx(boundary=(1, 0))
    z = _z([(0.5, 0.3), (0.0, 0.0)]).requires_grad_(True)
    e = guidance_energy(z, ctx, 0.0, 1.0, 0.0, 0.0)
    assert e.item() > 0.0
    g, = torch.autograd.grad(e, z)
    assert float(g[0, 0, 0]) > 0.0   # descent moves block 0 left (toward X0)
    # already at the left edge of the arrangement -> ~zero energy
    z2 = _z([(0.0, 0.3), (0.5, 0.0)])
    e2 = guidance_energy(z2, ctx, 0.0, 1.0, 0.0, 0.0)
    assert e2.item() < 1e-6


def test_hpwl_energy_pulls_connected_pair_together():
    ctx = _ctx(pairs=[(0, 1, 2.0)])
    z = _z([(0.0, 0.0), (0.8, 0.0)]).requires_grad_(True)
    e = guidance_energy(z, ctx, 0.0, 0.0, 1.0, 0.0)
    g, = torch.autograd.grad(e, z)
    assert float(g[0, 0, 0]) < 0.0   # descent moves block 0 toward block 1
    assert float(g[0, 1, 0]) > 0.0


def test_padding_blocks_contribute_nothing():
    ctx = _ctx()
    ctx.mask[0, 1] = False
    z = _z([(0.1, 0.1), (0.1, 0.1)]).requires_grad_(True)
    e = guidance_energy(z, ctx, 1.0, 1.0, 0.0, 0.0)
    assert e.item() < 1e-8            # only one real block: no overlap, boundary un-coded
```

- [ ] **Step 2: 跑測試確認 FAIL(ModuleNotFoundError)**

Run: `uv run pytest tests/test_partner_physics_guidance.py -q`

- [ ] **Step 3: 實作模組**

```python
"""Training-free physics-guided sampling: analytic energies over predicted
layouts (z-space), used by the Direct DDIM and Flow samplers.

Standalone by design: mirrors the xyaspect decode of
``diffusion_data.z_to_rectangles`` and the boundary-code semantics of
``my_opt_claude._viol_est`` (self-referential arrangement bbox), but does
not import either (z_to_rectangles is non-differentiable w.r.t. our use:
it hard-overwrites anchor channels).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class GuidanceConfig:
    k_steps: int = 5
    eta: float = 0.05
    t_gate: float = 0.35            # active when data_time >= 1 - t_gate
    w_overlap: float = 1.0
    w_boundary: float = 0.5
    w_hpwl: float = 0.1
    w_group: float = 0.0
    guide_aspect: bool = False
    trust_radius: float = 0.05      # max |Δz| per inner step (normalized units)

    @classmethod
    def from_env(cls) -> "GuidanceConfig":
        def f(name, default):
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default
        return cls(
            k_steps=int(f("PGUIDE_K", 5)),
            eta=f("PGUIDE_ETA", 0.05),
            t_gate=f("PGUIDE_TGATE", 0.35),
            w_overlap=f("PGUIDE_W_OVERLAP", 1.0),
            w_boundary=f("PGUIDE_W_BOUNDARY", 0.5),
            w_hpwl=f("PGUIDE_W_HPWL", 0.1),
            w_group=f("PGUIDE_W_GROUP", 0.0),
            guide_aspect=os.environ.get("PGUIDE_ASPECT") == "1",
            trust_radius=f("PGUIDE_TRUST", 0.05),
        )


@dataclass
class GuidanceContext:
    area: torch.Tensor           # [B,N] target areas (pad -> any; masked out)
    mask: torch.Tensor           # [B,N] bool
    known_mask: torch.Tensor     # [B,N,4] bool hard-anchor channels
    scale: torch.Tensor          # [B]
    boundary_code: torch.Tensor  # [B,N] long bitmask 1=L 2=R 4=T 8=B
    group_id: torch.Tensor       # [B,N] long (0 = none)
    pair_i: torch.Tensor         # [P] long
    pair_j: torch.Tensor         # [P] long
    pair_w: torch.Tensor         # [P] float

    def expand(self, batch: int) -> "GuidanceContext":
        e = lambda t: t.expand(batch, *t.shape[1:])
        return GuidanceContext(
            area=e(self.area), mask=e(self.mask), known_mask=e(self.known_mask),
            scale=self.scale.expand(batch), boundary_code=e(self.boundary_code),
            group_id=e(self.group_id),
            pair_i=self.pair_i, pair_j=self.pair_j, pair_w=self.pair_w,
        )


def rects_from_z(z: torch.Tensor, ctx: GuidanceContext):
    """Differentiable mirror of z_to_rectangles' xyaspect branch."""
    s = ctx.scale.view(-1, 1)
    x = z[..., 0] * s
    y = z[..., 1] * s
    area = torch.where(ctx.mask, ctx.area.clamp_min(1e-6),
                       torch.ones_like(ctx.area))
    aspect = torch.exp(z[..., 2].clamp(-3.0, 3.0))
    w = torch.sqrt(area * aspect).clamp_min(1e-6)
    h = torch.sqrt(area / aspect).clamp_min(1e-6)
    return x, y, w, h


def guidance_energy(z: torch.Tensor, ctx: GuidanceContext,
                    w_overlap: float, w_boundary: float,
                    w_hpwl: float, w_group: float) -> torch.Tensor:
    x, y, w, h = rects_from_z(z, ctx)
    m = ctx.mask.to(z.dtype)
    x1, y1 = x + w, y + h
    diag = ctx.scale.view(-1, 1).clamp_min(1.0)
    e = z.new_zeros(())

    if w_overlap:
        # signed-distance pairwise overlap area (soft-block differentiable)
        ox = (torch.minimum(x1.unsqueeze(2), x1.unsqueeze(1))
              - torch.maximum(x.unsqueeze(2), x.unsqueeze(1))).clamp_min(0.0)
        oy = (torch.minimum(y1.unsqueeze(2), y1.unsqueeze(1))
              - torch.maximum(y.unsqueeze(2), y.unsqueeze(1))).clamp_min(0.0)
        pm = m.unsqueeze(2) * m.unsqueeze(1)
        ov = ox * oy * pm
        ov = ov - torch.diag_embed(torch.diagonal(ov, dim1=1, dim2=2))
        e = e + w_overlap * 0.5 * ov.sum() / (diag.squeeze(1) ** 2).sum()

    if w_boundary and int(ctx.boundary_code.max()) > 0:
        # frame = arrangement bbox, detached: guidance moves blocks toward
        # the frame instead of collapsing the frame onto the block
        big = 10.0 * float(diag.max())
        X0 = torch.where(ctx.mask, x, torch.full_like(x, big)).min(dim=1, keepdim=True).values.detach()
        X1 = torch.where(ctx.mask, x1, torch.full_like(x1, -big)).max(dim=1, keepdim=True).values.detach()
        Y0 = torch.where(ctx.mask, y, torch.full_like(y, big)).min(dim=1, keepdim=True).values.detach()
        Y1 = torch.where(ctx.mask, y1, torch.full_like(y1, -big)).max(dim=1, keepdim=True).values.detach()
        code = ctx.boundary_code
        gb = z.new_zeros(x.shape)
        gb = gb + torch.where(code & 1 > 0, (x - X0).clamp_min(0.0), torch.zeros_like(x))
        gb = gb + torch.where(code & 2 > 0, (X1 - x1).clamp_min(0.0), torch.zeros_like(x))
        gb = gb + torch.where(code & 4 > 0, (Y1 - y1).clamp_min(0.0), torch.zeros_like(x))
        gb = gb + torch.where(code & 8 > 0, (y - Y0).clamp_min(0.0), torch.zeros_like(x))
        e = e + w_boundary * (gb * m / diag).sum() / m.sum().clamp_min(1.0)

    if w_hpwl and ctx.pair_i.numel():
        cx, cy = x + 0.5 * w, y + 0.5 * h
        dx = (cx[:, ctx.pair_i] - cx[:, ctx.pair_j]).abs()
        dy = (cy[:, ctx.pair_i] - cy[:, ctx.pair_j]).abs()
        e = e + w_hpwl * (ctx.pair_w * (dx + dy) / diag).sum() \
            / ctx.pair_w.sum().clamp_min(1e-6)

    if w_group and int(ctx.group_id.max()) > 0:
        cx, cy = x + 0.5 * w, y + 0.5 * h
        for gid in torch.unique(ctx.group_id):
            if int(gid) <= 0:
                continue
            gm = (ctx.group_id == gid) & ctx.mask
            if int(gm.sum()) < 2:
                continue
            gx = (cx * gm).sum(dim=1, keepdim=True) / gm.sum(dim=1, keepdim=True).clamp_min(1)
            gy = (cy * gm).sum(dim=1, keepdim=True) / gm.sum(dim=1, keepdim=True).clamp_min(1)
            e = e + w_group * (((cx - gx) ** 2 + (cy - gy) ** 2) * gm / diag ** 2).sum() \
                / gm.sum().clamp_min(1)
    return e
```

- [ ] **Step 4: 跑測試確認 PASS**

Run: `uv run pytest tests/test_partner_physics_guidance.py -q`
Expected: 5 passed。

- [ ] **Step 5: Commit**

```bash
git add src/solver/physics_guidance_claude.py tests/test_partner_physics_guidance.py
git commit -m "feat: add physics guidance energy core"
```

---

### Task 4: `guide_x0` 內迴圈(TDD)

**Files:**
- Modify: `src/solver/physics_guidance_claude.py`(追加)
- Test: `tests/test_partner_physics_guidance.py`(追加)

**Interfaces:**
- Produces: `guide_x0(z0, ctx, cfg, data_time) -> torch.Tensor`(同 shape;`data_time` ∈[0,1],1=clean 端)。
- Produces: `make_guidance(ctx, cfg) -> Callable[[torch.Tensor, float], torch.Tensor]`(給 sampler 的 closure)。

- [ ] **Step 1: 追加 failing tests**

```python
from physics_guidance_claude import GuidanceConfig, guide_x0, make_guidance


def test_guide_x0_reduces_overlap_energy():
    ctx = _ctx()
    cfg = GuidanceConfig(k_steps=8, eta=0.05, t_gate=0.5)
    z = _z([(0.1, 0.1), (0.1, 0.1)])
    e0 = guidance_energy(z, ctx, cfg.w_overlap, cfg.w_boundary, cfg.w_hpwl, cfg.w_group)
    z1 = guide_x0(z, ctx, cfg, data_time=0.9)
    e1 = guidance_energy(z1, ctx, cfg.w_overlap, cfg.w_boundary, cfg.w_hpwl, cfg.w_group)
    assert e1.item() < e0.item()


def test_guide_x0_identity_outside_t_gate():
    ctx = _ctx()
    cfg = GuidanceConfig(t_gate=0.35)
    z = _z([(0.1, 0.1), (0.1, 0.1)])
    out = guide_x0(z, ctx, cfg, data_time=0.2)   # 0.2 < 1-0.35
    assert torch.equal(out, z)


def test_guide_x0_freezes_known_and_aspect_channels():
    ctx = _ctx()
    ctx.known_mask[0, 0, :] = True               # block 0 fully anchored
    cfg = GuidanceConfig(k_steps=4, guide_aspect=False, t_gate=1.0)
    z = _z([(0.1, 0.1), (0.1, 0.1)], aspect=0.3)
    out = guide_x0(z, ctx, cfg, data_time=1.0)
    assert torch.equal(out[0, 0], z[0, 0])                  # anchored block untouched
    torch.testing.assert_close(out[..., 2], z[..., 2])      # aspect frozen
    assert not torch.equal(out[0, 1, :2], z[0, 1, :2])      # free block moved


def test_guide_x0_respects_trust_radius():
    ctx = _ctx()
    cfg = GuidanceConfig(k_steps=1, eta=100.0, trust_radius=0.03, t_gate=1.0)
    z = _z([(0.1, 0.1), (0.1, 0.1)])
    out = guide_x0(z, ctx, cfg, data_time=1.0)
    assert float((out - z).abs().max()) <= 0.03 + 1e-6
```

- [ ] **Step 2: 跑測試確認 FAIL**

Run: `uv run pytest tests/test_partner_physics_guidance.py -q`

- [ ] **Step 3: 實作(追加到模組)**

```python
def guide_x0(z0: torch.Tensor, ctx: GuidanceContext, cfg: GuidanceConfig,
             data_time: float) -> torch.Tensor:
    if data_time < 1.0 - cfg.t_gate or cfg.k_steps <= 0:
        return z0
    strength = data_time * data_time      # endpoint estimates reliable near clean end
    z = z0.detach()
    for _ in range(cfg.k_steps):
        z = z.detach().requires_grad_(True)
        e = guidance_energy(z, ctx, cfg.w_overlap, cfg.w_boundary,
                            cfg.w_hpwl, cfg.w_group)
        (g,) = torch.autograd.grad(e, z)
        g = g.masked_fill(ctx.known_mask, 0.0)
        if not cfg.guide_aspect:
            g = g.clone()
            g[..., 2:] = 0.0
        step = (cfg.eta * strength * g).clamp(-cfg.trust_radius, cfg.trust_radius)
        z = (z - step).detach()
        z = torch.cat([z[..., :2],
                       z[..., 2:3].clamp(-3.0, 3.0), z[..., 3:]], dim=-1)
        z = torch.where(ctx.known_mask, z0, z)
        z = z * ctx.mask.unsqueeze(-1)
    return z


def make_guidance(ctx: GuidanceContext, cfg: GuidanceConfig):
    def _fn(z0: torch.Tensor, data_time: float) -> torch.Tensor:
        return guide_x0(z0, ctx, cfg, data_time)
    return _fn
```

- [ ] **Step 4: 跑測試確認 PASS,Commit**

Run: `uv run pytest tests/test_partner_physics_guidance.py -q` → 9 passed

```bash
git add src/solver/physics_guidance_claude.py tests/test_partner_physics_guidance.py
git commit -m "feat: add guided x0 inner loop"
```

---

### Task 5: `sample_direct` guidance 注入 + steps 參數化(TDD)

**Files:**
- Modify: `src/solver/direct_model_claude.py:213-256`(`sample_direct`)
- Test: `tests/test_partner_physics_guidance.py`(追加)

**Interfaces:**
- Produces: `sample_direct(..., guidance: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None)`;`guidance=None` 時**逐行不動原代碼路徑**(bit-for-bit 等同)。
- Consumes: Task 4 的 closure 簽名 `(z0, data_time) -> z0`。

- [ ] **Step 1: 追加 failing tests(tiny 固定權重 denoiser)**

```python
from direct_model_claude import sample_direct


class _TinyDenoiser(torch.nn.Module):
    """Deterministic fake model: v = 0.1*z (no learned weights)."""
    def __init__(self):
        super().__init__()
        from types import SimpleNamespace
        self.config = SimpleNamespace(z_dim=4)

    def forward(self, z, t, node_feat, adj, mask, rel_feat=None, self_cond=None):
        return 0.1 * z * mask.unsqueeze(-1)


class _FakeSchedule:
    timesteps = 1000

    @staticmethod
    def alpha_sigma(t):
        tt = t.float() / 999.0
        alpha = torch.cos(tt * math.pi / 2).clamp(1e-4, 1.0).view(-1, 1, 1)
        sigma = torch.sin(tt * math.pi / 2).clamp(1e-4, 1.0).view(-1, 1, 1)
        return alpha, sigma


def _cond(n=3):
    return {
        "node_feat": torch.zeros(1, n, 2),
        "adj": None,
        "mask": torch.ones(1, n, dtype=torch.bool),
        "rel_feat": None,
    }


def test_sample_direct_none_guidance_matches_baseline():
    m, s = _TinyDenoiser(), _FakeSchedule()
    a = sample_direct(m, _cond(), s, steps=8,
                      generator=torch.Generator().manual_seed(3))
    b = sample_direct(m, _cond(), s, steps=8,
                      generator=torch.Generator().manual_seed(3), guidance=None)
    torch.testing.assert_close(a, b)


def test_sample_direct_guidance_changes_output_and_keeps_anchors():
    m, s = _TinyDenoiser(), _FakeSchedule()
    z_known = torch.zeros(1, 3, 4)
    z_known[0, 0] = torch.tensor([0.7, 0.7, 0.0, 0.0])
    known = torch.zeros(1, 3, 4, dtype=torch.bool)
    known[0, 0, :3] = True
    calls = []

    def shove(z0, data_time):
        calls.append(data_time)
        return z0 + 0.01

    base = sample_direct(m, _cond(), s, steps=8,
                         generator=torch.Generator().manual_seed(3),
                         z_known=z_known, known_mask=known)
    out = sample_direct(m, _cond(), s, steps=8,
                        generator=torch.Generator().manual_seed(3),
                        z_known=z_known, known_mask=known, guidance=shove)
    assert calls and all(0.0 <= dt <= 1.0 for dt in calls)
    assert not torch.equal(out, base)
    torch.testing.assert_close(out[0, 0, :3], z_known[0, 0, :3])  # anchors exact
```

- [ ] **Step 2: 跑測試確認 FAIL(unexpected keyword 'guidance')**

- [ ] **Step 3: 修改 `sample_direct`**

簽名加 `guidance=None`(放最後);迴圈內 L243-248 改為:

```python
        z0 = alpha * z - sigma * v
        z0[..., 2] = z0[..., 2].clamp(-3.0, 3.0)
        if z_known is not None and known_mask is not None:
            z0 = torch.where(known_mask, z_known, z0)
        if guidance is not None:
            data_time = 1.0 - float(t_val.item()) / max(schedule.timesteps - 1, 1)
            with torch.enable_grad():
                z0 = guidance(z0, data_time)
            z0[..., 2] = z0[..., 2].clamp(-3.0, 3.0)
            if z_known is not None and known_mask is not None:
                z0 = torch.where(known_mask, z_known, z0)
            sc = z0
            eps = (z - alpha * z0) / sigma.clamp_min(1e-6)
        else:
            sc = z0
            eps = sigma * z + alpha * v
```

(數學註:未 guided 時 `(z−α·z0)/σ ≡ σ·z+α·v`(α²+σ²=1);guided 時反推 eps 使 DDIM 下一步與 guided x̂₀ 一致 — MacroDiff+ Eq.12 的 v-pred 類比。`guidance=None` 分支保留原式,bit-for-bit 不變。)

- [ ] **Step 4: 跑測試(全檔 + 既有 flow/direct 測試)確認 PASS,Commit**

Run: `uv run pytest tests/test_partner_physics_guidance.py -q`

```bash
git add src/solver/direct_model_claude.py tests/test_partner_physics_guidance.py
git commit -m "feat: add optional x0 guidance hook to DDIM sampler"
```

---

### Task 6: my_opt 接線(`PARTNER_PHYSICS_GUIDE` / `PARTNER_DDIM_STEPS`)

**Files:**
- Modify: `src/solver/my_opt_claude.py:376-429`(`_sample_direct_raw_preds`)
- Modify: `src/solver/physics_guidance_claude.py`(追加 `build_context`)
- Test: `tests/test_partner_physics_guidance.py`(追加)

**Interfaces:**
- Produces: `build_context(at_d, cons_d, b2b, scale, known_mask) -> GuidanceContext`(`[1,N,*]` 張量;boundary/cluster 取 constraints 欄位 4/3,語義同 `legalizer_claude._parse_constraints`;b2b 上三角非零 → pair_i/j/w)。
- New env: `PARTNER_PHYSICS_GUIDE`(=1 啟用,預設 off)、`PARTNER_DDIM_STEPS`(預設 50)、`PGUIDE_*`(見 Task 3)。

- [ ] **Step 1: 追加 failing test**

```python
def test_build_context_parses_constraint_columns():
    from physics_guidance_claude import build_context
    n = 3
    at = torch.tensor([[100.0, 25.0, -1.0]])          # third = padding
    cons = torch.zeros(1, 3, 5)
    cons[0, 0, 4] = 1.0                                # boundary: left wall
    cons[0, 1, 3] = 2.0                                # cluster id 2
    b2b = torch.zeros(3, 3)
    b2b[0, 1] = 3.0
    scale = torch.sqrt(at.clamp_min(0).sum(1)).clamp_min(1.0)
    known = torch.zeros(1, 3, 4, dtype=torch.bool)
    ctx = build_context(at, cons, b2b, scale, known)
    assert bool(ctx.mask[0, 2]) is False
    assert int(ctx.boundary_code[0, 0]) == 1
    assert int(ctx.group_id[0, 1]) == 2
    assert ctx.pair_i.tolist() == [0] and ctx.pair_j.tolist() == [1]
    assert abs(float(ctx.pair_w[0]) - 3.0) < 1e-6
```

- [ ] **Step 2: 實作 `build_context`(追加到 physics_guidance_claude.py)**

```python
def build_context(area_target: torch.Tensor, constraints: torch.Tensor,
                  b2b: torch.Tensor, scale: torch.Tensor,
                  known_mask: torch.Tensor) -> GuidanceContext:
    mask = area_target > 0
    c = constraints
    ncol = c.shape[-1]
    boundary = c[..., 4].round().long() if ncol > 4 else torch.zeros_like(mask, dtype=torch.long)
    group = c[..., 3].round().long() if ncol > 3 else torch.zeros_like(mask, dtype=torch.long)
    n = area_target.shape[1]
    w = b2b[:n, :n]
    iu = torch.triu_indices(n, n, offset=1)
    vals = w[iu[0], iu[1]] + w[iu[1], iu[0]]
    nz = vals > 0
    return GuidanceContext(
        area=area_target, mask=mask, known_mask=known_mask,
        scale=scale.reshape(-1), boundary_code=boundary, group_id=group,
        pair_i=iu[0][nz], pair_j=iu[1][nz], pair_w=vals[nz].float(),
    )
```

- [ ] **Step 3: 接線 `_sample_direct_raw_preds`**

L417-421 改為:

```python
            guide = None
            if os.environ.get("PARTNER_PHYSICS_GUIDE") == "1":
                from physics_guidance_claude import (GuidanceConfig,
                                                     build_context,
                                                     make_guidance)
                ctx = build_context(at_d, cons_d, b2b.to(dev), scale,
                                    known).expand(K_s)
                guide = make_guidance(ctx, GuidanceConfig.from_env())
            z = sample_direct(self.direct_model, cond_k,
                              self.direct_schedule,
                              steps=_env_int("PARTNER_DDIM_STEPS", 50),
                              generator=gen,
                              z_known=z_known.expand(K_s, -1, -1),
                              known_mask=known.expand(K_s, -1, -1),
                              guidance=guide)
```

- [ ] **Step 4: 跑全部相關測試 + 介面驗證**

Run: `uv run pytest tests/test_partner_physics_guidance.py -q && bash scripts/validate.sh`
Expected: 測試 PASS;validate(bare defaults,guidance off)通過 = 現役行為未變。

- [ ] **Step 5: Commit**

```bash
git add src/solver/my_opt_claude.py src/solver/physics_guidance_claude.py tests/test_partner_physics_guidance.py
git commit -m "feat: wire opt-in physics guidance into direct sampling"
```

---

### Task 7: Guidance 快 probe(尾段案 K/η 掃描)— 接線完成後立即跑

**Files:**
- Create: `scripts/probes/pguide_quick_probe.py`
- Output: `artifacts/pguide/quick_probe.json`

**Interfaces:**
- Consumes: `MyOptimizer._sample_direct_raw_preds`(env-gated guidance)與 `my_opt_claude._constraint_penalties`/overlap 計數作評分。
- Produces: 每 (case, config) 的 raw 候選統計:pairwise overlap 總面積、boundary 違規估計(`_viol_est`)、best-of-K prescreen 分,及 GPU 採樣 wall-clock。

- [ ] **Step 1: 寫 probe(骨架,直接可跑)**

```python
"""Quick physics-guidance probe: raw candidate quality vs (K, eta) on
tail/boundary-dense cases. No refine, no golden labels; ~2 min on GPU."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "partner"))

CASES = [0, 49, 79, 88, 93, 96, 99]
GRID = [dict(k=0), dict(k=3, eta=0.05), dict(k=5, eta=0.05),
        dict(k=10, eta=0.05), dict(k=5, eta=0.1)]


def main():
    import numpy as np
    import torch
    from my_opt_claude import MyOptimizer
    from case_loader_claude import load_case      # official LiteTensorDataTest loader

    out = []
    opt = MyOptimizer()
    for cid in CASES:
        n, at, cons, tpos, b2b, p2b, pins = load_case(cid)
        for cfgi in GRID:
            os.environ["PARTNER_PHYSICS_GUIDE"] = "1" if cfgi["k"] else "0"
            os.environ["PGUIDE_K"] = str(cfgi.get("k", 0))
            os.environ["PGUIDE_ETA"] = str(cfgi.get("eta", 0.05))
            t0 = time.time()
            preds = opt._sample_direct_raw_preds(n, at, cons, tpos, b2b, p2b,
                                                 pins, K=15)
            torch.cuda.synchronize()
            dt = time.time() - t0
            ov = [float(_overlap_area(p, n)) for p in preds]
            pen = opt._constraint_penalties(preds, n, at, cons) or [0.0] * len(preds)
            out.append(dict(case=cid, **cfgi, sample_s=dt,
                            overlap_median=float(np.median(ov)),
                            overlap_best=float(np.min(ov)),
                            viol_median=float(np.median(pen)),
                            viol_best=float(np.min(pen))))
            print(out[-1])
    Path(ROOT / "artifacts/pguide").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "artifacts/pguide/quick_probe.json", "w"), indent=1)


def _overlap_area(P, n):
    import numpy as np
    x0, y0 = P[:n, 0], P[:n, 1]
    x1, y1 = x0 + P[:n, 2], y0 + P[:n, 3]
    ox = np.clip(np.minimum(x1[:, None], x1[None]) - np.maximum(x0[:, None], x0[None]), 0, None)
    oy = np.clip(np.minimum(y1[:, None], y1[None]) - np.maximum(y0[:, None], y0[None]), 0, None)
    m = ox * oy
    return (m.sum() - np.trace(m)) / 2.0


if __name__ == "__main__":
    main()
```

注意:`case_loader_claude.load_case` 若不存在,改用 `scripts/probes/retrieval_trace.py` 的既有 case 載入函式(同 LiteTensorDataTest 路徑);probe 撰寫者先 `grep -n "LiteTensorDataTest" scripts/probes/*.py` 對齊現行 loader。`PARTNER_PRESCREEN_V=1` 需設定使 `_constraint_penalties` 生效。

- [ ] **Step 2: 跑 probe**

Run: `PARTNER_PRESCREEN_V=1 uv run python scripts/probes/pguide_quick_probe.py`
Expected: k>0 配置在 overlap_median / viol_median 上顯著低於 k=0;sample_s 增量 < 1s/config。

- [ ] **Step 3: 判準(gate to Task 9)**

- overlap_best 降幅 ≥30% 且 viol 不升 → 帶最佳 (K,η) 進 Task 9 full-100。
- 無改善或 sample_s 爆炸(>2× 基線)→ 調 t_gate/trust_radius 重掃一次;再無效 → guidance 記為 disproven-at-this-config,不進 full-100。

- [ ] **Step 4: Commit**

```bash
git add scripts/probes/pguide_quick_probe.py
git commit -m "feat: add physics guidance quick probe"
```

---

### Task 8: Budget 定案 + POOL 對照 + DDIM-25 probe(依賴 T1 結果)

**Files:**
- Create: `scripts/probes/analyze_budget_scan.py`(判讀)
- Output: `artifacts/partner_eval/budget_scan_summary.md`、後續 runs JSON

- [ ] **Step 1: 判讀掃描(腳本算 Q-time 彈性 + alpha 投影)**

```python
"""Summarize budget scan: no-runtime Q per MAX + alpha-referenced projection.
Reuses the projection math from the 0723 analysis (deadline-bounded: official
runtime = budget)."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MED = {int(r["test_id"]): float(r["median_runtime_s"]) for r in csv.DictReader(
    open(ROOT / "docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv"))}


def total(path):
    d = json.load(open(path))
    lam = [math.exp(r["block_count"] / 12) for r in d["test_results"]]
    s = sum(lam)
    proj = 0.0
    for r, l in zip(d["test_results"], lam):
        q = 10.0 if not r["is_feasible"] else r["cost_no_runtime"]
        rtf = r["runtime_seconds"] / MED[r["test_id"]]
        proj += (l / s) * min(q * max(0.7, rtf ** 0.3), 10 - 1e-6)
    rt = sum(r["runtime_seconds"] for r in d["test_results"])
    return d["total_score_no_runtime"], proj, rt


print(f"{'config':28s} {'noRT_Q':>8s} {'alphaProj':>10s} {'sum_rt':>8s}")
for name in ["cont_retrieval_direct_control", "budget_scan_max8",
             "budget_scan_max6", "budget_scan_max4"]:
    p = ROOT / f"artifacts/partner_eval/{name}.json"
    if p.exists():
        q, proj, rt = total(p)
        print(f"{name:28s} {q:8.4f} {proj:10.4f} {rt:7.0f}s")
```

- [ ] **Step 2: 定案準則**

- 選 **alphaProj 最低**的 MAX;若相鄰檔差 <0.01 取較保守(較大 MAX)。
- 若 MAX=4 的 noRT_Q 突跳(>+0.03 vs MAX=6)→ 標記 direct cliff,鎖定 MAX=6。
- 產出:定案 env 組(`PARTNER_BUDGET_MAX=<X>`),寫入 `budget_scan_summary.md`。

- [ ] **Step 3: 在定案 budget 下跑兩個追加對照(序跑背景)**

```bash
# POOL 46(品質補償):
PARTNER_BUDGET_MAX=<X> PARTNER_POOL=46 <計分env> ... --output artifacts/partner_eval/budget<X>_pool46.json
# DDIM 25 步(GPU 減步;吐出的時間留在 budget 窗內給 SA/refine):
PARTNER_BUDGET_MAX=<X> PARTNER_DDIM_STEPS=25 <計分env> ... --output artifacts/partner_eval/budget<X>_ddim25.json
```

(完整命令同 Task 1 腳本樣式。)判準:noRT_Q 相對定案 budget run 改善或持平即收案(POOL 46 預期 −0.005~−0.01;DDIM25 允許 ≤+0.002 換取 direct 下限減半,供 Task 9/13 組合)。

- [ ] **Step 4: Commit 判讀腳本 + summary**

```bash
git add scripts/probes/analyze_budget_scan.py artifacts/partner_eval/budget_scan_summary.md
git commit -m "docs: budget scan verdict and follow-up runs"
```

---

### Task 9: Guidance full-100 gate(依賴 T7 最佳 config + T8 定案 budget)

- [ ] **Step 1: 對照組(guide off,定案 budget)= Task 8 的定案 run,不重跑**
- [ ] **Step 2: Treatment**

```bash
PARTNER_BUDGET_MAX=<X> PARTNER_PHYSICS_GUIDE=1 PGUIDE_K=<K*> PGUIDE_ETA=<η*> \
<計分env> ... --output artifacts/partner_eval/budget<X>_pguide.json
```

- [ ] **Step 3: 判準(evaluator-evidence 促轉紀律)**

- `total_score_no_runtime` 改善(目標 ≥−0.005)**且** per-case runtime 不升(guidance 藏在 GPU 窗;p90/max 差 <2%)→ 促轉,env 進 `.env` 提案。
- 改善集中於 boundary 稠密/尾段案 → 依 reusable stats 加開 `PGUIDE_W_BOUNDARY` 自適應(單獨再 gate)。
- 無改善 → 記 disproven(附 probe/full-100 證據),guidance 保持 off。

- [ ] **Step 4: 寫實驗記錄 + Commit**

```bash
git add docs/experiments/2026-07-2X-physics-guidance-gate.md
git commit -m "docs: physics guidance full-100 gate verdict"
```

---

### Task 10: Flow my_opt 整合(gate plan Task 4;立即可做,overfit ckpt 驗證)

**Files:**
- Modify: `src/solver/my_opt_claude.py`
- Test: `tests/test_partner_flow_integration.py`(new)

**Interfaces:**
- New env: `FLOW_CKPT`(路徑,預設空=off)、`PARTNER_FLOW_SLOTS`(預設 0)、`PARTNER_FLOW_STEPS`(預設 8)、`PARTNER_FLOW_SOLVER`(預設 euler)。
- Produces: `_sample_flow_preds(self, n, at, cons, tpos, b2b, p2b, pins, K) -> List[np.ndarray]`;與 Direct 候選合併走**同一** `rank_predictions` 與**不變的** refine 容量(replace-not-add,沿用 `candidate_supply_claude.allocate_quotas`)。
- Consumes: worktree 的 `flow_matching_claude.sample_flow`(修過 S-1)與 `flow_train_claude.checkpoint_method`(先把兩檔從 worktree 同步進 main repo `partner/`,見 Step 0)。

- [ ] **Step 0: 同步 F1/F2 模組進 main repo(不含訓練中 checkpoint)**

```bash
cp /nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/src/solver/flow_matching_claude.py partner/
cp /nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/src/solver/flow_train_claude.py partner/
cp /nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/tests/test_partner_flow_matching.py tests/
uv run pytest tests/test_partner_flow_matching.py -q   # 5 passed
```

- [ ] **Step 1: failing tests(quota 沿用/checkpoint 守門/預設關閉)**

```python
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "partner"))

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


def test_flow_env_defaults_off():
    assert os.environ.get("FLOW_CKPT") is None
    from my_opt_claude import MyOptimizer
    opt = MyOptimizer.__new__(MyOptimizer)      # no heavy init
    assert getattr(opt, "flow_model", None) is None
```

- [ ] **Step 2: 實作 loader + `_sample_flow_preds`(my_opt 內,樣式沿 `_sample_direct_raw_preds`)**

```python
    def _load_flow_model(self):
        self.flow_model = None
        path = os.environ.get("FLOW_CKPT")
        slots = _env_int("PARTNER_FLOW_SLOTS", 0)
        if not path or slots <= 0 or not Path(path).exists():
            return
        try:
            from flow_train_claude import checkpoint_method
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            checkpoint_method(ckpt)
            from direct_model_claude import DirectDenoiser, DirectConfig
            cfg = DirectConfig(**ckpt["model_config"])
            model = DirectDenoiser(cfg).to(self.device)
            model.load_state_dict(ckpt.get("ema") or ckpt["model"])
            model.eval()
            self.flow_model = model
            self.flow_cfg = cfg
        except Exception:
            self.flow_model = None    # flow failure never harms Direct

    def _sample_flow_preds(self, n, at, cons, tpos, b2b, p2b, pins, K):
        from direct_train_claude import fast_condition
        from direct_model_claude import known_z_channels
        from flow_matching_claude import sample_flow
        dev = self.device
        at_d = at.unsqueeze(0).to(dev)
        cons_d = cons.unsqueeze(0).to(dev)
        tpos_d = tpos.unsqueeze(0).to(dev)
        cond = fast_condition(
            at_d, b2b.unsqueeze(0).to(dev), p2b.unsqueeze(0).to(dev),
            pins.unsqueeze(0).to(dev), cons_d, tpos_d,
            relation_feat_dim=self.flow_cfg.relation_feat_dim,
            node_feat_dim=self.flow_cfg.node_feat_dim)
        scale = layout_scale(at_d)
        z_known, known = known_z_channels(at_d, cons_d, tpos_d, scale)
        gen = torch.Generator(device=dev)
        gen.manual_seed(23)
        cond_k = {k: (v.expand(K, *v.shape[1:]).contiguous()
                      if torch.is_tensor(v) else v) for k, v in cond.items()}
        with torch.no_grad():
            sample = sample_flow(
                self.flow_model, cond_k,
                steps=_env_int("PARTNER_FLOW_STEPS", 8),
                solver=os.environ.get("PARTNER_FLOW_SOLVER", "euler"),
                generator=gen,
                z_known=z_known.expand(K, -1, -1),
                known_mask=known.expand(K, -1, -1))
            rects = z_to_rectangles(
                sample.z, at_d.expand(K, -1),
                target_positions=tpos_d.expand(K, -1, -1),
                constraints=cons_d.expand(K, -1, -1),
                z_repr=self.flow_cfg.z_repr)
        return [rects[k, :n].cpu().numpy().astype(np.float64) for k in range(K)]
```

接線點:在 Direct 候選生成處(`_sample_direct_raw_preds` 的呼叫端)以 `allocate_quotas(refine_total, {"direct": refine_total - flow_slots, "flow": flow_slots}, ...)` 分配、合併後共 rank、截斷到原 refine 容量(**refine 總數不變**)。

- [ ] **Step 3: 跑測試 + 用 overfit ckpt 煙測整合路徑**

```bash
uv run pytest tests/test_partner_flow_integration.py -q
FLOW_CKPT=/nashome/NVL4/vdalab/yyds-dev/codex-worktrees/flow-matching-f1-f3/checkpoints/flow_matching_overfit/final.pt \
PARTNER_FLOW_SLOTS=3 bash scripts/eval_single.sh 21   # 單案跑通、feasible 即可(品質不預期)
```

- [ ] **Step 4: Commit**

```bash
git add src/solver/flow_matching_claude.py src/solver/flow_train_claude.py src/solver/my_opt_claude.py tests/
git commit -m "feat: add opt-in flow candidate source (replace-not-add)"
```

**正式 gate(v1 訓完後才跑,非本 task 範圍)**:D control / F only / D+F 各 full-100,判準沿 gate plan F4 + 「同品質下採樣時間」維度。

---

### Task 11: guided-FM(`sample_flow` 加 guidance 參數)

**Files:**
- Modify: `src/solver/flow_matching_claude.py`(main repo 副本;worktree 訓練端不動)
- Test: `tests/test_partner_flow_matching.py`(追加)

**Interfaces:**
- Produces: `sample_flow(..., guidance=None)`;在 `v0 = velocity(z, t0, ...)` 之後、Euler/Heun step 之前:

```python
        if guidance is not None and t0 > 0.0:
            z0_hat = z + (1.0 - t0) * v0
            with torch.enable_grad():
                z0_hat = guidance(z0_hat, t0)      # data_time = t0 (FM: t=1 clean)
            v0 = (z0_hat - z) / max(1.0 - t0, 1e-3)
```

- Consumes: Task 4 `make_guidance` closure(同一介面)。

- [ ] **Step 1: failing test(guidance 被呼叫、None 等價、known 通道不受汙染)** — 測試樣式同 Task 5 的兩則,把 `sample_direct` 換成 `sample_flow`(ConstantVelocity 假模型沿用該檔既有 fixture)。
- [ ] **Step 2: 實作(上述片段;`guidance=None` 分支代碼不動)**
- [ ] **Step 3: 跑測試 PASS + Commit**

```bash
git add src/solver/flow_matching_claude.py tests/test_partner_flow_matching.py
git commit -m "feat: add endpoint guidance hook to flow sampler"
```

(v1 訓完後的 guided-FM-{8,16} vs guided-DDIM probe 沿 Task 7 骨架擴 `--flow-checkpoint`,判準:quality-per-GPU-second。)

---

### Task 12: numba SA 內迴圈(profile 先行;獨立軌)

**Files:**
- Create: `scripts/probes/profile_partner_sa.py`;視 profile 結果 Modify `src/solver/legalizer_claude.py`
- Test: `tests/test_partner_sa_numba_equiv.py`(new)

- [ ] **Step 1: Profile 單案(n=120)找熱點**

```bash
uv run python -m cProfile -s cumtime -o /tmp/sa_profile.pstats \
  scripts/eval_single.sh 用的 evaluator 路徑對 case 99 …    # 或:
uv run python - <<'EOF'
import cProfile, pstats, sys
sys.path.insert(0, "partner")
# 直接呼叫 legalizer 的單 restart 入口(_layout / anneal)於 case 99 輸入
# (實作者:grep -n "def _anneal\|def _layout" src/solver/legalizer_claude.py 取入口)
EOF
```

判準:找出佔 SA worker ≥60% cumtime 的葉函式(預期:column packing cost 評估 / 鄰域 move 評分)。

- [ ] **Step 2: numba 化該葉函式(`@numba.njit(cache=True)`,輸入改為 ndarray 簽名)**

模板(以 cost 迴圈為例;實際函式以 profile 為準):

```python
from numba import njit

@njit(cache=True)
def _pack_cost_nb(order, widths, heights, col_break, frame_w):
    # 逐欄裝箱 + 高度累加,回傳 (total_h, viol) — 與原 Python 逐行同義
    ...
```

- [ ] **Step 3: 等價測試(同輸入,numba vs python 輸出 bit-equal)+ throughput 測試(iters/s ≥2×)**
- [ ] **Step 4: 在定案 budget 下跑 full-100 對照(noRT_Q 應改善 ~−0.005 級;runtime 不變 — deadline-bounded)**
- [ ] **Step 5: Commit**

```bash
git add src/solver/legalizer_claude.py scripts/probes/profile_partner_sa.py tests/test_partner_sa_numba_equiv.py
git commit -m "perf: numba-accelerate SA inner loop"
```

---

### Task 13: 組合定案 full-100 + Beta 打包演練

**Files:**
- Create: `submission/op_wrapper.py`、`submission/requirements.txt`、`scripts/package_beta.sh`

- [ ] **Step 1: 組合定案 run**(T8 定案 budget × T9 過 gate 的 guidance × T8 收案的 POOL/DDIM25 × T12 numba)全開一次 full-100;`total_score_no_runtime` + alpha 投影都須 ≥ 各單項最佳(無負交互)。
- [ ] **Step 2: `op_wrapper.py`**(flat 目錄,相對路徑載 checkpoint;env 旗標以 `os.environ.setdefault` 內嵌定案值 — 評測機不 source 我們的 .env):

```python
"""ICCAD 2026 beta submission entry point (cadc1013)."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for k, v in {
    "DIRECT_CKPT": str(HERE / "checkpoints" / "direct_v2_final.pt"),
    "VKILL_OFF": "1", "PARTNER_PRESCREEN_V": "1", "PARTNER_NREF": "15",
    "PARTNER_OVERSAMPLE": "4", "PARTNER_TAG_ANCHOR_EXTRA": "3",
    # Task 8/9/12 定案值填此(budget MAX / POOL / guidance / steps):
    "PARTNER_BUDGET_MAX": "<X>", "PARTNER_POOL": "<P>",
}.items():
    os.environ.setdefault(k, v)

from my_opt_claude import MyOptimizer as ContestOptimizer  # noqa: E402

MyOptimizer = ContestOptimizer
```

- [ ] **Step 3: `requirements.txt`**:先試 **Case A(空檔)**——contest 環境已提供 numpy/torch/scipy/numba/shapely/threadpoolctl;打包演練若 import 失敗才轉 Case B(完整凍結 `uv pip freeze` 裁剪)。
- [ ] **Step 4: 打包演練**:tar 出 `cadc1013.tar.gz` → 解壓到乾淨目錄 → `python iccad2026_evaluate.py --evaluate op_wrapper.py` 跑全 100(合規檢查:flat、無絕對路徑、檔名精確、無多餘 .py/log/結果檔)。
- [ ] **Step 5: Commit + 提交**

```bash
git add submission/ scripts/package_beta.sh
git commit -m "feat: beta submission package"
```

---

## Self-Review 記錄

- Spec 覆蓋:physics_guidance(T3-7,9)✓ budget 掃描定案(T1,8)✓ 品質補償雙件套 POOL(T8)/numba(T12)✓ GPU 減步(T8 DDIM25 + T6 env 化)✓ flow S-1(T2)✓ Task4 整合(T10)✓ guided-FM(T11)✓ beta 打包(T13)✓。
- 型別一致:`make_guidance` closure `(z0, data_time)->z0` 在 T4 定義、T5/T11 消費一致;`GuidanceContext.expand` 在 T6 消費;`allocate_quotas` 簽名沿 `candidate_supply_claude` 既有。
- 已知不確定:T7 的 `load_case` loader 名需實作者對齊現行 probe(標注在 task 內);T12 熱點函式待 profile(模板已給);T13 `<X>/<P>` 由 T8/T9 結果填入。
