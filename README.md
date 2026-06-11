# Data-Driven SoC Floorplanning

本專案是 ICCAD 2026 FloorSet Challenge Problem C 的 SoC floorplanning solver。當前主線是 `Architecture v4`：用 Anchor-GNN checkpoint 產生 block-level 幾何先驗，再由 `floorset_arch` 的 relative-order / hetero-graph constructive decoder 生成合法 placement，最後用 repair pass 修正 overlap、boundary、grouping 與 MIB 等限制。

目前 production 預設 checkpoint：

```bash
checkpoints/gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.pt
```

若要覆蓋模型，可以設定 `.env` 或環境變數：

```bash
FLOORSET_GNN_CHECKPOINT=checkpoints/your_checkpoint.pt
```

## Problem 定義

FloorSet Problem C 要求對一組 SoC blocks 產生 2D floorplan。每個 block 必須輸出 `(x, y, w, h)`，其中 `(x, y)` 是左下角座標，`w/h` 是寬高。

輸入包含：

- `block_count`：有效 block 數量。
- `area_targets`：每個 soft block 的目標面積。
- `b2b_connectivity`：block-to-block 加權連線，用於 HPWL。
- `p2b_connectivity` 與 `pins_pos`：block-to-terminal 加權連線與 terminal 座標。
- `constraints`：固定形狀、預放置、MIB、grouping、boundary 等限制。
- `target_positions`：訓練資料或 evaluator 提供的參考位置，對 fixed/preplaced block 特別重要。

硬限制：

- 所有 blocks 不可重疊。
- soft block 面積誤差必須在 1% 以內。
- fixed-shape block 必須保留指定尺寸。
- preplaced block 必須保留指定位置與尺寸。
- 違反硬限制的 case 會被視為 infeasible，單 case cost 會被固定懲罰為 `M = 10`。

軟限制：

- `Boundary`：指定 block 應貼齊 bounding box 的邊或角。
- `Grouping`：同 group blocks 應透過邊相鄰形成連通 component。
- `MIB`：同 MIB group blocks 應有一致的 width/height。
- 軟限制違反不會讓解 infeasible，但會進入指數懲罰項。

目標函數核心是同時降低 HPWL gap、bounding-box area gap、soft violation ratio 與 runtime factor。分數越低越好，100 個 validation/test cases 會按 block count 指數加權，所以大型 case 的失敗會主導 total score。

## Total Score No Runtime

`total_score_no_runtime` 是本 repo 目前做架構比較與 checkpoint promotion 的主要本地指標。它沿用官方 evaluator 的 quality 與 soft-violation 計算，但把 runtime adjustment 固定為 `1.0`：

```text
cost_no_runtime = quality_factor * violation_factor
```

用途：

- 比較 solver 架構、repair policy、candidate matrix 與 checkpoint 時，先排除本地 runtime median 的扭曲。
- 保留 `total_score` 與 raw runtime 作為 runtime risk signal，但不讓 WSL/local median artifact 成為唯一決策依據。
- checkpoint promotion 以 full validation 的 `total_score_no_runtime` 為準，不只看 supervised validation loss。

相關文件：

- `CONTEXT.md`：定義 No-Runtime Quality Score、Local Runtime-Aware Score、Production Solver Path 等專案語言。
- `docs/optimization-notes.md`：記錄目前調參政策與不要重複嘗試的 ablation。
- `docs/evaluation/2026-05-11-no-runtime-promotion.md`：記錄 no-runtime checkpoint promotion 與 large-case matrix evidence。

## Clone 與安裝

第一次 clone repo 時要把 submodule 一起拉下來，否則 `FloorSet/` 會是 empty 或缺少官方 evaluator / dataset files。

```bash
git clone --recursive <repo-url>
cd Data-Driven-SoC-Floorplanning
```

如果已經 clone 過但 `FloorSet/` 是空的，或忘記加 `--recursive`：

```bash
git submodule update --init --recursive
```

```bash
bash scripts/install.sh
```

這會初始化 `FloorSet` submodule、安裝 `uv`、建立 venv、安裝 dependencies，並用官方 template 跑一個 quick evaluator smoke test。

## Scripts 目的與使用方法

### `scripts/iccad2026_evaluate.py`

這是本 repo 保存的 no-runtime 版本 evaluator。因為 `FloorSet/` 是官方 repo/submodule，這個專案無法直接把修改後的 `iccad2026_evaluate.py` push 回官方 repo，所以第一次 clone 或每次重置 submodule 後，請手動 copy 這個版本去取代官方 submodule 裡的 evaluator：

```bash
cp scripts/iccad2026_evaluate.py FloorSet/iccad2026contest/iccad2026_evaluate.py
```

取代後，`scripts/eval_total.sh` 才會穩定輸出：

```text
Total Score (No Runtime)
Avg Cost (No Runtime)
```

如果之後執行 `git submodule update --init --recursive`、切換 `FloorSet` 版本，或 submodule 被 reset，請重新執行一次 copy 指令。

### `scripts/eval_single.sh`

跑單一 validation/test case，預設 `test_id=95`。會讀取 `.env`，若未設定 `FLOORSET_GNN_CHECKPOINT`，會使用目前 production 預設 checkpoint。

```bash
bash scripts/eval_single.sh
bash scripts/eval_single.sh 99
```

### `scripts/eval_total.sh`

跑完整 100-case evaluator，並輸出 `Total Score`、`Total Score (No Runtime)`、平均 runtime、median runtime、P90 runtime 等指標。

```bash
bash scripts/eval_total.sh
bash scripts/eval_total.sh gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.pt
bash scripts/eval_total.sh checkpoints/gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.pt
bash scripts/eval_total.sh /abs/path/to/checkpoint.pt
```

checkpoint 解析規則：

- 無參數：優先用 `.env` 的 `FLOORSET_GNN_CHECKPOINT`，否則用既有 shell 環境變數，再否則用 production 預設 checkpoint。
- 傳入 checkpoint 參數時：該參數優先於 `.env` 與既有 shell 環境變數。
- 只有檔名：視為 `checkpoints/<name>`。
- repo-relative path：視為 repo root 下的相對路徑。
- absolute path：直接使用。

### `scripts/validate.sh`

執行官方 submission interface validate，不跑完整 scoring：

```bash
bash scripts/validate.sh
```

### `scripts/train.sh`

訓練 Anchor-GNN checkpoint。預設參數與目前 v4 方向一致：`hidden_dim=192`、`layers=6`、`num_samples=800000`、`epochs=4`、`lr=3e-4`、`val_samples=10000`、pairwise head 開啟，並使用 weighted dirty-sample training。

```bash
bash scripts/train.sh
```

常用 override：

```bash
NUM_SAMPLES=200000 EPOCHS=10 LR=4e-4 bash scripts/train.sh
OUTPUT_DIR=checkpoints CHECKPOINT_TAG=remote_run bash scripts/train.sh
WANDB_MODE=offline WANDB=0 bash scripts/train.sh
RESUME_CHECKPOINT=checkpoints/old.pt bash scripts/train.sh
```

重要訓練政策：

- `CLEAN_SAMPLE_POLICY=weighted` 是預設值，因為目前 clean-only sample ratio 太低。
- dirty samples 仍可提供低權重 geometry reference。
- dirty order/pairwise supervision 預設被抑制，避免把 soft-violating `fp_sol` 當成可靠 constraint oracle。
- `WRITE_STABLE_CHECKPOINTS=1` 才會覆寫穩定檔名，例如 `gnn_best.pt`。

### `scripts/train_transformer.sh`

訓練 Graph Transformer encoder 版本的 Anchor-GNN。參數格式與 `scripts/train.sh` 相同，但預設 `ENCODER=graph-transformer`、`NUM_HEADS=8`、`CHECKPOINT_PREFIX=gnn_transformer`，log 檔名也會帶 `train_arch_v5_transformer`。

```bash
bash scripts/train_transformer.sh
DEVICE=cpu WANDB=0 NUM_SAMPLES=2 VAL_SAMPLES=1 EPOCHS=1 HIDDEN_DIM=16 LAYERS=1 bash scripts/train_transformer.sh
OUTPUT_DIR=checkpoints CHECKPOINT_TAG=remote_transformer_run bash scripts/train_transformer.sh
RESUME_CHECKPOINT=checkpoints/old_transformer.pt bash scripts/train_transformer.sh
```

### `scripts/train_hgt.sh`

訓練 Local HGT encoder 版本的 Anchor-GNN。HGT 會保留 block、pin、cluster、MIB 與 boundary typed nodes，並只沿 heterogeneous factor graph 的 typed local edges 做 relation-specific attention；v1 不加入 global attention/refinement layer。參數格式與 `scripts/train_transformer.sh` 相同，但預設 `ENCODER=hgt`、`NUM_HEADS=4`、`BATCH_SIZE=8`、`CHECKPOINT_PREFIX=gnn_hgt`，log 檔名會帶 `train_arch_v5_hgt`。HGT script 也預設對高風險樣本啟用低風險 decoder-aware ranking 權重：`HIGH_RISK_ORDER_MULTIPLIER=1.5`、`HIGH_RISK_PAIRWISE_MULTIPLIER=1.5`，只加強 order/pairwise 訓練 loss，不改 decoder/repair。

```bash
bash scripts/train_hgt.sh
DEVICE=cpu WANDB=0 NUM_SAMPLES=2 VAL_SAMPLES=1 EPOCHS=1 HIDDEN_DIM=16 LAYERS=1 bash scripts/train_hgt.sh
BATCH_SIZE=4 bash scripts/train_hgt.sh
OUTPUT_DIR=checkpoints CHECKPOINT_TAG=remote_hgt_run bash scripts/train_hgt.sh
RESUME_CHECKPOINT=checkpoints/old_hgt.pt bash scripts/train_hgt.sh
```

HGT dirty-sample experiments can enable repaired pseudo targets:

```bash
ENABLE_REPAIRED_PSEUDO_TARGETS=1 bash scripts/train_hgt.sh
DIRTY_PSEUDO_ORDER_WEIGHT=0.20 DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT=0.35 ENABLE_REPAIRED_PSEUDO_TARGETS=1 bash scripts/train_hgt.sh
```

Guidance ablations are evaluator-time knobs:

```bash
FLOORSET_GUIDANCE_ANCHOR_ONLY=1 bash scripts/eval_total.sh checkpoints/model.pt
FLOORSET_GUIDANCE_DISABLE_PAIRWISE=1 bash scripts/eval_total.sh checkpoints/model.pt
```

### `scripts/update.sh`

自動 `fetch/pull/status/add/commit/push`，預設推到 `origin arch-v4`。適合個人工作流，但在 reviewer 會檢查輸出時建議先手動看 diff。

```bash
bash scripts/update.sh "update: message"
```

## `src/` 模組架構

```text
src/
├── architecture_v5_optimizer.py
└── floorset_arch/
    ├── __init__.py
    ├── budget_layer.py
    ├── constructive.py
    ├── diagnostics.py
    ├── features.py
    ├── geometry.py
    ├── hetero_graph.py
    ├── models.py
    ├── optimizer.py
    ├── parser.py
    ├── quality_portfolio.py
    ├── relative_order.py
    ├── repair.py
    ├── risk_budget.py
    ├── scoring.py
    ├── surrogate_guidance.py
    ├── v10_proxy.py
    ├── nn/
    │   ├── __init__.py
    │   └── model.py
    └── training/
        ├── __init__.py
        ├── checkpoint.py
        ├── losses.py
        ├── promote_checkpoint.py
        ├── pseudo_targets.py
        ├── selection.py
        └── train.py
```

### Active production path

`src/architecture_v5_optimizer.py` 是目前 evaluator scripts 使用的 contest-facing wrapper。它把 repo root、`src/` 和 `FloorSet/iccad2026contest` 加到 `sys.path`，並把 `MyOptimizer` / `ContestOptimizer` 指向 `floorset_arch.optimizer.ArchitectureV5Optimizer`。

`floorset_arch.optimizer.ArchitectureV4Optimizer` 目前保留為 `ArchitectureV5Optimizer` 的相容 alias，供舊 tests 或歷史 docs 讀懂演進脈絡；新的 eval/validate script 一律使用 v5 wrapper。

`src/floorset_arch/optimizer.py` 是 production solver 入口。`solve()` 的流程是：

1. `parser.parse_instance()` 把 evaluator tensors 轉成 `Instance`。
2. `_try_anchor_guidance()` 載入 `FLOORSET_GNN_CHECKPOINT` 或預設 checkpoint，產生 `AnchorGuidance`。
3. `_candidate_specs()` 依 instance statistics 選 candidate profiles。
4. candidate 由 `relative_order.construct_relative_order_placement()` 或 opt-in `constructive.construct_beam_placement()` 生成。
5. `repair.repair_placement()` 修正 overlap、boundary、grouping、MIB。
6. `_select_best_candidate()` 用 soft-first ranking 與 no-runtime proxy 選出 best placement。
7. 回傳 evaluator 要求的 `(x, y, w, h)` list。

### Core files

- `models.py`：核心資料模型，包含 `Rect`、`Placement`、`SolverConfig`、`AnchorGuidance`、`Instance`。目前預設 checkpoint 在 `SolverConfig.default_checkpoint`。
- `parser.py`：清理 evaluator tensors，建立 fixed/preplaced、boundary、MIB、cluster、connectivity lookup。
- `features.py`：建立 Anchor-GNN 使用的 node features 與 edge tensors。
- `nn/model.py`：`FloorplanGNN`，輸出 anchor、priority、aspect 與 pairwise relation logits。
- `relative_order.py`：目前主要 constructive decoder。使用 GNN anchor、connectivity、constraint statistics 決定 block shape、order 與相對位置。
- `constructive.py`：MER/skyline slot beam decoder。保留為 opt-in candidate 與幾何策略來源。
- `hetero_graph.py`：建立 block、pin、cluster、MIB、boundary 的 hetero/factor-style graph，讓 constraints 成為顯式 solver input。
- `repair.py`：後處理 repair，包括 hard snap、overlap relocation、boundary snap、cluster connection、MIB shape unification、large-case opt-in repair。
- `scoring.py`：local HPWL proxy、candidate scoring 與 placement scoring。
- `diagnostics.py`：placement metrics、soft violation counts、repair delta，用於 trace 與 ablation。
- `training/train.py`：Anchor-GNN training CLI。
- `training/losses.py`：anchor target、order/pairwise loss、clean/dirty sample weighting。
- `training/checkpoint.py`：checkpoint save/load 與 run tag。

### Legacy/reference files

- `FloorSet/iccad2026contest/iccad2026_evaluate.py`：官方 evaluator，本 repo 已使用 `time.perf_counter()` 計時，並支援 `total_score_no_runtime` 輸出。

## Repository File Tree

```text
.
├── README.md
├── CONTEXT.md
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── install.sh
│   ├── validate.sh
│   ├── eval_single.sh
│   ├── eval_total.sh
│   ├── iccad2026_evaluate.py
│   ├── train.sh
│   └── update.sh
├── src/
│   ├── architecture_v5_optimizer.py
│   └── floorset_arch/
├── tests/
│   ├── test_optimizer.py
│   ├── test_constructive.py
│   ├── test_relative_order.py
│   ├── test_scoring.py
│   ├── test_geometry.py
│   ├── test_repair.py
│   ├── test_parser.py
│   ├── test_diagnostics.py
│   └── test_evaluator_scoring.py
├── checkpoints/
│   ├── gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.pt
│   └── other experiment checkpoints
├── docs/
│   ├── optimization-notes.md
│   ├── evaluation/
│   └── superpowers/
├── problem/
│   ├── C_floorset_challenge_zh.md
│   ├── C_floorset_challenge_keypoint.md
│   └── official PDFs / QA PDFs
└── FloorSet/
    ├── iccad2026contest/
    ├── LiteTensorDataTest/
    ├── floorset_lite/
    └── dataset loaders and utilities
```

## 最近更新

- README 已改成中文專案入口文件，補上 problem 定義、no-runtime scoring、scripts、`src/` 架構、逐檔功能與 file tree。
- Production 預設 checkpoint 已更新為 `checkpoints/gnn_best_0519_ns1000000_ep3_encmpnn_h256_l6_acc32.pt`。
- `scripts/eval_single.sh`、`scripts/eval_total.sh` 與 `SolverConfig.default_checkpoint` 已同步使用新的 best checkpoint。
- `tests/test_optimizer.py` 的預設 checkpoint 測試已同步更新。

## 測試

```bash
uv run pytest
```

本次 checkpoint 預設更新已跑過聚焦測試：

```bash
uv run pytest tests/test_optimizer.py
```
