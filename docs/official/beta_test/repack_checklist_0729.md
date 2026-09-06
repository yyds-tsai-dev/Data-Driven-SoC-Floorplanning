# 2026-07-29 beta 提交包重打（cadc1013）— spec 對照 checklist

對照 `docs/official/beta_test/beta_submission_guidelines_problemC.txt`（違反即
DISQUALIFICATION）。本次重打取代 0728 舊包，理由有二：**舊包混入未使用的 .py（清潔度違規）**、
且 **舊包裝錯 direct checkpoint（step 1139000，非 0729 定案證據所用的 step 1200000）**。

> **注意：現役包是 §7 的 0730 新檔名版（md5 `a80b90d3…`）。以下 §1-§5 描述 0729 舊檔名版，判準與結論仍適用。**

- 包（0729 版，已被 §7 取代）：`submission/cadc1013_0729_oldnames.tar.gz.bak`（799,222,482 bytes，md5 `2bf39655bd42e458170078539c0712c8`；
  檔案模式統一 644 / 目錄 755，`--owner=0 --group=0 --numeric-owner`）
- 內容 root：`cadc1013/`（扁平），13 個 .py + `requirements.txt`(0 bytes) + `checkpoints/` 2 個 .pt
- 來源：repo branch `5.6-sol-reduce-time`，partner 模組取自本次 commit 的工作樹（含 0729 promote 的
  dpmpp10 + refine stall-stop）

## 1. spec 條款 × 包內狀態

| spec 條款 | 要求 | 包內狀態 | 判定 |
| --- | --- | --- | --- |
| §1 archive 命名 | `cadc<team_id>.tar.gz` | `cadc1013.tar.gz` | PASS |
| §1 解開為扁平目錄 | `cadc1013/` | `tar tzf` 全部條目以 `cadc1013/` 開頭，無外層包裹 | PASS |
| §1/§3 `op_wrapper.py` | 必要、精確檔名、頂層 | `cadc1013/op_wrapper.py`（唯一入口） | PASS |
| §1/§3 `op_src.py` | 選用 | 不提供（提供等同再放一份 optimizer，與「不得重複 .py」相衝） | N/A |
| §1 `requirements.txt` | 必要，可為空 | 頂層，**0 bytes**（Case A：只用預裝 numpy/torch/scipy/numba/tqdm/shapely/threadpoolctl） | PASS |
| §1 no nesting | 三個必要檔不得在子目錄 | 三檔皆在 `cadc1013/` 頂層 | PASS |
| §1 helper 可放子目錄但用相對路徑 | — | helper .py 全在頂層；權重在 `checkpoints/`，由 `op_wrapper.py` 以 `Path(__file__).resolve().parent` 推導 | PASS |
| §1 不得含無關檔 | 無 log/結果 JSON/scratch/其他嘗試 | 無 `.log`/`.json`/`.ipynb`/`__pycache__`/隱藏檔（打包前建立、打包後才演練） | PASS |
| §1 不得有多餘或重複 .py | 恰一份 op_wrapper，無其他 optimizer .py | 13 個 .py 全部在 `op_wrapper.py` 的實際 import 閉包內（見 §2），無第二個 optimizer 入口 | PASS |
| §1 未使用的大型二進位須移除 | — | 只有 2 個 .pt，皆在定案 env 下實際載入（見 §3 正向驗證） | PASS |
| §1/§4b 不得絕對路徑 | — | `grep -rnE "/nashome|/home/|/tmp/|yyds-dev|codex-worktrees"` 於包內 .py = 0 命中；無任何字串常數形式的絕對路徑 | PASS |
| §2 requirements 完整性 | 非空則須含傳遞依賴 | 空檔（Case A），不觸發 fresh venv 安裝 | PASS |
| §5 評測指令 | `python iccad2026_evaluate.py --evaluate op_wrapper.py`（cwd=cadc1013/） | 已在解壓後的乾淨目錄實跑（見 §4） | PASS |
| §5 optimizer 須為 FloorplanOptimizer 子類 | — | `my_opt_claude.MyOptimizer(FloorplanOptimizer)`，由 op_wrapper 以 `MyOptimizer`/`ContestOptimizer` 曝露 | PASS |
| §6 100 案不 crash | — | **R1 已補**：從最終 tar 全新解壓跑 full-100（官方 evaluator + repo evaluator 兩 pass）：noRT 1.1283 / proj 0.8736 / **100/100 feasible** / tailQ 1.1163，與 refstall_on 家族（1.1253-1.1265）差 0.002-0.003 在判準內；direct step 1200000 + flow step 1000000 均載入無靜默失敗（artifacts/partner_eval/repack_full100_verify.json，0729 13:0x，GPU 有 v3 訓練爭用） | PASS |

## 2. .py 閉包：留 13 / 刪 6

留（皆為 `op_wrapper.py` 在**定案 env** 下的實際 import 閉包）：

`op_wrapper.py`、`my_opt_claude.py`、`candidate_supply_claude.py`、`diffusion_data.py`、
`diffusion_model.py`、`direct_model_claude.py`、`direct_train_claude.py`、`flow_matching_claude.py`、
`flow_train_claude.py`、`legalizer_claude.py`、`noise_opt_claude.py`、`physics_guidance_claude.py`、
`refiner_claude.py`

其中三個「看起來像訓練腳本」但**運行必需**，非「其他嘗試過的做法」：

- `direct_train_claude.py` — `fast_condition`（Direct/flow 兩通道的 condition builder，每案必經）
- `flow_train_claude.py` — `checkpoint_method`（flow checkpoint 的 tag 驗證，載入時必經）
- `noise_opt_claude.py` + `physics_guidance_claude.py` — 定案 `PARTNER_FLOW_ANTITHETIC=1` 走
  `noise_opt_claude.sample_flow_diff`（`my_opt_claude._sample_flow_preds`），而 `noise_opt_claude`
  頂層 import `physics_guidance_claude`。**兩者是承重件，不是判死通道的殘留。**

刪（6 個，皆為定案 env 下不可達）：

| 檔 | 為何可刪 | 處理 |
| --- | --- | --- |
| `vkill_claude.py` | `VKILL_OFF=1` 且未設 `VKILL`；import 本就在 gate 之後的函式內 | 直接不打包 |
| `retrieval_index_claude.py` | 需 `PARTNER_RETRIEVAL_INDEX`+`PARTNER_RETRIEVAL_SLOTS`，皆未設；本就是函式內 import | 直接不打包 |
| `retrieval_features_claude.py`、`retrieval_matching_claude.py`、`retrieval_transfer_claude.py` | 同上，但原本是 `my_opt_claude` 頂層 import | **lazy 化**後不打包 |
| `diffusion_train.py` | 只提供 `known_target_positions_from_fp`，僅 `train_step` 用得到 | **lazy 化**後不打包 |

lazy-import 化（只搬 import 位置，不改任何功能行為）：

- `src/solver/my_opt_claude.py`：刪頂層 L41-43 三個 `retrieval_*` import，改到
  `_sample_retrieval_preds` 內第一個 `try:` 區塊（在 `self.retrieval_index is None` 早退之後，
  且 ImportError 會被既有的 `except Exception` 收斂成空 batch）。
- `src/solver/direct_train_claude.py`：`from diffusion_train import known_target_positions_from_fp`
  由頂層搬進 `train_step`。
- `tests/test_partner_retrieval_integration.py`：spy 從 `my_opt_claude.remap_boundary_node_features`
  改掛到來源模組 `retrieval_transfer_claude`（lazy import 在呼叫時才綁定，patch 來源模組才生效）。
- 回歸：`uv run pytest tests/ -q -k partner` → **175 passed**。

## 3. checkpoint 身分驗證

比對用 EMA state-dict 的 md5（依 tensor 名排序、float32 bytes）。

| 檔 | 來源 | step | EMA hash | 判定 |
| --- | --- | --- | --- | --- |
| 舊包 `direct_v2_final.pt` | 不明（非 1.2M） | 1139000 | `2c1788fa…` | **不符 → 汰換** |
| 新包 `direct_v2_final.pt` | `partner/checkpoints/direct_v2_cont/eval_step1p2M.pt` | 1200000 | `49241d84…` | 與來源逐 tensor 一致 |
| 舊/新包 `flow_matching_v1_final.pt` | flow worktree `checkpoints/flow_matching_v1/final.pt` | 1000000 | `4e9a890d…` | 與來源一致，沿用不動 |

打包時只保留推論端會讀的鍵（`model_config`/`args`/`step`/`ema`/`model`，其中 `model` 與 `ema`
指向同一組張量——`my_opt_claude._load_direct_model` 先 `load_state_dict(ck["model"])` 再逐 tensor
以 `ck["ema"]` 覆寫，故等價），丟掉 `optimizer`/`sched` 訓練狀態：1.72 GB → 430 MB/檔。
flow 端 `ckpt.get("ema") or ckpt["model"]` 同樣取 EMA。

## 4. 從 tar 全新演練

`tar -xzf` 到乾淨臨時目錄 → 複製官方 `iccad2026_evaluate.py` 進 `cadc1013/`（模擬評測端）→
`PYTHONPATH` 只給 FloorSet 資料/loader（**不含 repo 的 `partner/`**，缺檔會真的炸）→
用 repo venv 模擬預裝清單（requirements.txt 為 0 bytes，不建 fresh venv）。

**通道正向驗證**（必要：flow 載入失敗是靜默的，evaluator 跑得動 ≠ 包完整）：
`direct_model loaded: True (step 1200000)`、`flow_model loaded: True (step 1000000)`、device `cuda`、
`fast_condition`/`sample_direct_dpmpp`/`sample_flow`/`sample_flow_diff` 四個 lazy import 全部解析成功、
被刪的 4 個模組確認在此組態下不可 import 也不被 import。

**官方指令實跑**（`python iccad2026_evaluate.py --evaluate op_wrapper.py`，官方 evaluator，
RuntimeFactor=1.0 中性）：case 0 → 1.3799 / 0.61 s、case 43 → 1.4901 / 1.41 s、
case 95 → 1.0781 / 3.40 s，3/3 feasible。

**與定案基準對照**（同一解壓包，改用 repo 的 `scripts/iccad2026_evaluate.py` 以取得可比的
no-runtime 欄位；基準 = `artifacts/partner_eval/refstall_on_rep3.json` 同案）：

| test_id | n | 演練 no-RT | 基準 no-RT | Δ | 演練 runtime | 基準 runtime | feasible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 21 | 1.4231 | 1.4785 | −3.7% | 0.60 s | 0.61 s | yes |
| 43 | 64 | 1.4941 | 1.4837 | +0.7% | 1.41 s | 1.41 s | yes |
| 95 | 116 | 1.0648 | 1.0451 | +1.9% | 3.21 s | 3.17 s | yes |

三案皆在 ±10% 內（差異量級與單案 SA 隨機性一致），runtime 與基準幾乎重合。
演練期間 GPU 另有 flow v3 訓練佔用（已知條件）。

上表演練跑在權限正規化前的同內容包上；權限正規化後的**最終 tar** 另做一次全新解壓複驗
（case 95：官方 evaluator → 1.0648 / 3.43 s / feasible，direct step 1200000 + flow step 1000000 皆載入）。

## 5. 定案 env（`op_wrapper.py` 的 `os.environ.setdefault`）

相對 0728 版三處變更（證據：`docs/experiments/2026-07-29-dpmpp-sampler-and-flow-fewstep.md`）：

- `PARTNER_DDIM_STEPS` 25 → **10**
- 新增 `PARTNER_DIRECT_SOLVER=dpmpp`
- 新增 `PARTNER_REFINE_STALL_STOP=1`

其餘不變：`VKILL_OFF=1`、`PARTNER_PRESCREEN_V=1`、`PARTNER_NREF=15`、`PARTNER_OVERSAMPLE=4`、
`PARTNER_TAG_ANCHOR_EXTRA=3`、`PARTNER_BUDGET_MAX=3.5`、`PARTNER_DIRECT_MIN=2.0`、
`PARTNER_FLOW_SOLVER=euler`、`PARTNER_FLOW_SLOTS=10`、`PARTNER_FLOW_STEPS=8`、
`PARTNER_FLOW_ANTITHETIC=1`、`DIRECT_CKPT`/`FLOW_CKPT` 相對推導。

全部用 `setdefault`：評測端若自行注入同名變數，其值優先。

## 6. 殘留風險

- **R1（中）**：本次只跑 3 案抽驗，未在解壓包上跑完整 100 案（§6 checklist 最後一項只到 PARTIAL）。
  同版程式碼在 repo 內 full-100 已 9 runs 100/100 feasible，但那是從 `partner/` 直跑、
  而非從包內跑。提交前建議補一次「從解壓包跑 full-100」。
- **R2（低）**：`direct_train_claude.py` 頂層 `from iccad2026_evaluate import FloorplanDatasetLite,
  get_training_dataloader, train_floorplan_collate`——依賴評測端 evaluator 仍 re-export 這三個名字
  （官方與 repo 兩版目前都有，L49-51）。若官方 beta 版 evaluator 拿掉，Direct 通道會整條啞掉且
  部分路徑是靜默的。可加固的做法是把 `fast_condition` 抽成不依賴 evaluator 的獨立模組。
- **R3（低）**：`noise_opt_claude.py` / `physics_guidance_claude.py` 名稱看起來像「試過的做法」，
  但確為 `PARTNER_FLOW_ANTITHETIC=1` 的承重件；若評測端以檔名做人工清潔度判讀可能被誤判。
  本文件 §2 即為書面理由。
- **R4（低）**：`my_opt_claude` 的 v1 diffusion 種子 checkpoint（`DEFAULT_CHECKPOINT`）在包內不存在，
  載入時印一行 `checkpoint not found ... using heuristic init only`——這與所有證據 run 的行為一致
  （repo 內同樣沒有該檔），非退化。
- **R5（低）**：archive 799 MB，全部是兩個 430 MB 權重；若主辦對單隊大小有未公布的上限會有風險。

## 7. 0730 新檔名重打（現役包；覆蓋 §1-§5 的檔名與 md5）

`partner/` 模組在 `d9aa665` 改為語義檔名、`a11273f`/`c84e26d` 收割 flow branch 之後，
提交包依同一份 §1-§4 判準重打。**閉包成員、定案 env、兩顆 checkpoint 全部不變，只換檔名。**

- 包：`submission/cadc1013.tar.gz`（**799,227,068 bytes**，md5 **`a80b90d3714b2e026b4bcc84a41b5237`**；
  模式 644/755、uid/gid 0/0）
- 舊包備份：`submission/cadc1013_0729_oldnames.tar.gz.bak`（md5 `2bf39655…`）＋
  `submission/cadc1013_0729_oldnames/`（皆 gitignored）
- checkpoint 直接沿用已驗證的兩顆（未重新產生）：
  `checkpoints/direct_v2_final.pt` md5 `15133794ee2afc9f1289eb168241f824`（step 1200000）、
  `checkpoints/flow_matching_v1_final.pt` md5 `7d207b32a1d764c2d8dff0e116389860`（step 1000000）

### 模組名對照（13 檔閉包）

| 舊名（0729 包） | 新名（0730 包） |
| --- | --- |
| `op_wrapper.py` | `op_wrapper.py`（唯一改動：`from contest_optimizer import MyOptimizer`） |
| `my_opt_claude.py` | `contest_optimizer.py` |
| `legalizer_claude.py` | `column_sa_legalizer.py` |
| `refiner_claude.py` | `layout_refiner.py` |
| `direct_model_claude.py` | `direct_diffusion_model.py` |
| `direct_train_claude.py` | `direct_diffusion_train.py` |
| `flow_matching_claude.py` | `flow_matching_model.py` |
| `flow_train_claude.py` | `flow_matching_train.py` |
| `noise_opt_claude.py` | `noise_optimization.py` |
| `physics_guidance_claude.py` | `physics_guidance.py` |
| `candidate_supply_claude.py` | `candidate_supply.py` |
| `diffusion_data.py` / `diffusion_model.py` | 不變 |

不打包（新名）：`violation_killer`、`retrieval_{index,features,matching,transfer}`、`diffusion_train`、
`frame_repack`、`analytic_polish`、`direct_diffusion_train_v2`、`flow_matching_distill`
（後四者本來就不在 0729 包內）。全部十個都以實測確認：在定案 env 下既不被 import，也不可 import。

### 稽核

- 絕對路徑掃描 0 命中；13 個 .py，無多餘/重複 optimizer .py、無 `__pycache__`/log/JSON；
  `tar tzf` 全部條目在 `cadc1013/` 之下；`requirements.txt` 0 bytes。
- 12 個 helper 與 `partner/` 對應檔**逐 byte 相同**（`op_wrapper.py` 為包專屬）。
- flow 收割未動 runtime：`flow_matching_model.py` 相對 `dee4156` 是**純新增**（0 行刪除），
  `sample_flow` / `endpoint_from_velocity` / `flow_path` 三個 runtime 函式 AST 逐節點相同；
  `direct_diffusion_model.py` 與 `direct_model_claude.py` 逐 byte 相同；
  `flow_matching_train.FLOW_METHODS` 含 `flow_matching_v1`，現役 checkpoint 照常通過 tag 驗證。

### 從新 tar 全新解壓 → full-100（GPU 無其他佔用，v3 訓練已結束）

| pass | 指令 | total | noRT | feasible | errors | avg / max runtime |
| --- | --- | --- | --- | --- | --- | --- |
| 官方 evaluator | `python iccad2026_evaluate.py --evaluate op_wrapper.py` | 1.1391 (RF=1.0) | — | **100/100** | 0 | 1.99 s / 3.55 s |
| repo evaluator | 同上，取 no-runtime 欄位 | 1.3030 | **1.1354** | **100/100** | 0 | 1.99 s / 3.52 s |

`artifacts/partner_eval/repack_full100_newnames_{official,repo}.json`。overlap 違規總數 0，
tailQ（n>100 平均 noRT）1.1286。載入訊息確認：
`loaded direct model step 1200000` + `loaded flow model step 1000000`（新模組名下），
兩通道 `direct_model`/`flow_model` 皆非 None、device `cuda`，無靜默失敗。

noRT 1.1354 對 0729 R1 的 1.1283 為 **+0.0071**（判準 ≤0.01 內），但確實落在
refstall_on 家族觀測帶（1.1253-1.1265）之上——依上述 byte/AST 同一性證據，這是抽樣變異
而非改名或收割造成的退化；若要收窄不確定性，唯一乾淨做法是同期成對再跑一輪。

## 8. 0810 Beta Resubmission 重打(現役包;回應官方 format issue 通知)

官方 0810 email:Beta Submission 有 format issue(s),8/12 23:59 (GMT+8) 前重傳到
Google Drive 的 **「Beta Resubmission」資料夾**。對照 guidelines 稽核 0806 定版包
(md5 `08c00842…`),找到三處違規,全部修正後重打;**26 個 .py 閉包、兩顆
checkpoint、op_wrapper 定案 env 完全不動**(僅 op_wrapper 一行註解更新)。

### 0806 包的三個 format 問題

| # | 問題 | 違反條款 | 修正 |
| --- | --- | --- | --- |
| 1 | `__pycache__/` 整目錄入包(43 檔:.pyc + numba .nbi/.nbc,0806「烘焙 JIT 快取」的刻意決定) | §1 清潔度(「Do NOT include files unrelated…may lead to disqualification」) | 整目錄移除。冷 JIT 已由 op_wrapper import-time `_warm_jit_kernels()` 覆蓋(QA A14 載入不計時),快取非必要 |
| 2 | `requirements.txt` 非空且不完整(僅 `scipy==1.13.1`+`numba==0.66.0`) | §2/§4a:非空=Case B,評測端**只用此檔建 fresh venv** → 無 torch/numpy,整包必炸;而 §2 Case A 明文 scipy、numba 皆預裝 | 改回 **0 bytes**(Case A)。0806 依據的「官方 requirements 無 scipy/numba」是 contest repo 的 requirements.txt,對 beta 評測環境不適用 |
| 3 | 權限/擁有者未正規化(`retrieval_*.py` mode 600 — 評測端異 uid 讀不到;owner 洩漏 `yyds-dev`;模式 644/664 混雜) | 非明文條款,但 0729/0730 版既有標準 | `find` 統一 644/755,tar `--owner=0 --group=0 --numeric-owner --sort=name` |

### 第四個修正(Codex 獨立稽核 round 1 抓到):`contest_optimizer.py` → `op_src.py`

Codex(gpt-5.4)對重打包的稽核在其餘條款全 PASS 之下,判 §1
「No other optimizer .py files」嚴格解讀 FAIL:`contest_optimizer.py` 檔名帶
optimizer 且持有唯一的 `MyOptimizer(FloorplanOptimizer)` 子類,嚴格審查者可能
視為第二個 optimizer 檔。修法=改名為官方許可的 **`op_src.py`**(§1/§3 明文
選用檔名,兼作 op_wrapper 失敗時的官方 fallback 入口),`op_wrapper.py` L68
import 同步改 `from op_src import MyOptimizer`。程式引用僅此一處
(其餘 `contest_optimizer` 字樣皆為註解/docstring,保持與 `partner/` 逐 byte
一致故不動)。附帶收益:`--evaluate op_src.py` 獨立實測 case 0/95 皆 feasible
(無 env defaults 時走大預算 ~0.6/19.5s,緊急備援語義,品質較差但可動)。

### 定版

- 包:`submission/cadc1013.tar.gz`(md5
  **`c2dd42bbf042af4d29d24c4b014a7cb6`**);31 條目 = 26 .py(含 `op_src.py`)
  + requirements.txt(0B)+ 2 .pt + 2 目錄,無 `__pycache__`/log/JSON/隱藏檔;
  絕對路徑掃描 0 命中;模式 644/755、uid/gid 0/0。
- 0806 舊包備份:`submission/cadc1013_0806.tar.gz.bak`(md5 `08c00842…`)。
- op_wrapper.py 相對 0806 僅兩處變更:L68 import 改 `op_src`、numba 來源註解
  改為「provided by the evaluation environment (Beta guidelines §2 Case A)」。
- 上傳注意:**只傳 `cadc1013.tar.gz` 一個檔**到「Beta Resubmission」資料夾
  (勿附 .md5 sidecar 或其他檔案)。

### 從新 tar 全新解壓 → full-100(§6 鐵律)

全新解壓到臨時目錄、複製官方 `iccad2026_evaluate.py` 進 `cadc1013/`、
`PYTHONPATH` 只給 `FloorSet/`(loader+資料,不含 repo `partner/`)、repo venv
模擬預裝清單(numba 0.66.0 / scipy 1.13.1 / torch 2.6.0+cu124,量測窗乾淨:
GPU idle、無他人 evaluator),官方指令
`python iccad2026_evaluate.py --evaluate op_wrapper.py`,兩輪:

| 包 | Total(RF=1.0) | feasible | errors | avg / max rt | 首案 rt |
| --- | --- | --- | --- | --- | --- |
| 改名前(md5 `6182b7e6…`) | 1.1673 | 100/100 | 0 | 0.295 / 1.797 s | 0.085 s |
| **定版(md5 `c2dd42bb…`)** | **1.1647** | **100/100** | **0** | **0.293 / 1.791 s** | **0.083 s** |

- 首案無冷 JIT 懸崖——證明移除烘焙快取後 op_wrapper import-time warm 足夠;
  `loaded direct model step 1200000` + `loaded flow model step 1000000`,
  `[polish]` 逐案觸發可見(scipy 通道活著)。
- 對 0806 證據帶 1.1692:Δ −0.002~−0.005,遠小於此檔位家族單跑 sd
  (≈0.008-0.010),與「僅移除垃圾檔+改名、零行為變更」一致。

### 本次一併修好的 repo 問題

`d9aa665` 是純 `git mv`（0 行變更），因此被改名模組**內部的 import 敘述仍指向舊 `*_claude` 名**
（`contest_optimizer.py` 一檔 29 處），`c84e26d` 這個 HEAD 事實上 import 不起來；
`606 passed` 反映的是工作樹而非 HEAD。本次把該跟進修正（8 檔）一併提交，
提交包即建構於此修正之上。
