# 2026-09-06 repo 整理:只留最終提交路徑

賽後整理(deadline 2026-08-31 已過)。目標:repo 內的 source code 只剩「最終提交包 cadc1013 的來源」與其訓練/打包/驗證工具;實驗記錄全部留在 `docs/`;暫存檔與被否決的實驗碼移除(git 歷史仍可找回,見下方 commit)。

## 最終提交(對照)

| 包 | md5 | 內容 | 演練(乾淨 py3.13 venv、官方 evaluator) |
|---|---|---|---|
| B′ `submission/FINAL_UPLOAD/cadc1013.tar.gz` | `3f2cda42…` | FT2 flow prior(round-2 T=12 250k EMA)+ FLOW_SLOTS 16 / NREF 12 + legal guard 面積修復 + mid 預算表 | 1.0858、100/100、avg 0.363 s、max 1.31 s |
| C′ `submission/FINAL_UPLOAD_C/cadc1013.tar.gz` | `6d0ca94e…` | B′ + FT2×FT3 soup α0.5 flow prior(唯一 diff = flow ckpt + op_wrapper 一行) | 1.0834、100/100 |

08-31 05:20 UTC 上傳 B′;14:47 UTC 給出 C′ 上傳綠燈。**實際最終上傳為 B′ 或 C′ 待使用者確認**(`docs/project-status/2026-08-31-final-handoff.md`)。工作樹目前是 C′ 設定;`bash scripts/pack_cadc1013.sh <dir>` 重建出的包與 C′ 逐檔 `diff -r` 完全相同(整理當日驗證)。回 B′:`bash scripts/release/apply_pack_variant.sh ft2 s16`。

## Source code 搬移(commit `55e4316`)

| 舊 | 新 | 說明 |
|---|---|---|
| `partner/<26 個 closure 模組>.py`、`partner/contest_optimizer.py` | `src/solver/` | 出貨包的 import closure;檔名 = 包內模組名,內容 byte-identical |
| `tests/synth_instances.py` | `src/solver/synth_instances.py` | 隨包出貨(op_wrapper JIT 暖機) |
| `partner/shipping/` | `src/shipping/` | `op_wrapper.py`(包內 env setdefaults)、`requirements.txt`、`budget_table_mid.txt` |
| `partner/icdc/` | `src/icdc_engine/` | IC/DC engine + topology-prior trainer(產出出貨的 `direct_v2_student_s2.pt`);外部 import `icdc.` → `icdc_engine.` |
| `partner/stage_receipt.py` | `src/solver/stage_receipt.py` | probe-only 收據,`tests/test_postpass_router.py` 仍用 |
| `tests/test_partner_*` / `tests/test_icdc_*` | `tests/test_solver_*` / `tests/test_icdc_engine_*` | |
| `scripts/build_partner_retrieval_index.py`、`scripts/probes/profile_partner_sa.py` | `scripts/build_retrieval_index.py`、`scripts/probes/profile_solver_sa.py` | |
| `scratchpad/flow_ft_0828/*.py`、`run_ft.sh`;`artifacts/flow_ft_08{29,30,30b}/run_ft.sh` | `scripts/training/flow_finetune/` | 出貨 flow prior 的 fine-tune 來源(round 1–4) |
| `scratchpad/rtaware/{apply_pack_variant.sh,dryrun12.sh,rt_pairs.py,rt_total.py}` | `scripts/release/{apply_pack_variant.sh,rehearse_package.sh,runtime_aware_pairs.py,runtime_aware_total.py}` | 決賽日打包變體 / 演練 / runtime-aware 分析 |
| `scratchpad/flow_ft_08{28,29,30}/STATUS.md`、`scratchpad/legalizer_fork_gap_0729.md` | `docs/experiments/flow-finetune/`、`docs/experiments/2026-07-29-legalizer-fork-gap.md` | 文件歸 docs |

## 移除(commit `83e8022` + 本 commit)

- `src/floorset_arch/`、`src/architecture_v{5,11}_optimizer.py`(2026-05~07 Anchor-GNN / diffusion / column-backbone 路線)、其 37 個測試、`scripts/train*.sh`、`promote_checkpoint.sh`、`diffusion_diagnostic.py`、`wandb_sidecar.py`、PyInstaller 路徑(`frozen_entry.py`、`smoke_frozen.sh`、`package_submission.sh`、`scripts/op_wrapper.py`)、`update.sh`。
- 64 個 LFS 追蹤的 legacy checkpoints(bytes 留在 `artifacts/legacy_floorset_arch_checkpoints/`,不再追蹤)。
- closure 以外、gate 否決的 partner 模組:`analytic_polish`、`constructive_g01`、`eplace_arm`、`frame_reinsert`、`flow_matching_distill`、`harness/`,及其測試 / probes;依賴已拆除的 direct_v2 續訓評測迴路的 probe runner(`partner_eval_cont.sh`、`run_retrieval_gate.sh`、`run_budget_scan.sh`、`run_flow_variant_eval.sh`、`run_noiseopt_eval.sh`)。
- `scratchpad/` 全部(271 個追蹤檔 + 未追蹤 chain/log;`soup_ft2ft3_a50.pt` 與 `submission/cadc1013/checkpoints/flow_matching_ft0831_soupa50_250k_ema.pt` md5 相同,`a25` 為否決變體)。docs 內對 `scratchpad/...` 的引用是歷史紀錄,對應工具若仍需要見上表的 `scripts/release/`。

## `artifacts/`(git-ignored,只改磁碟目錄名)

| 舊 | 新 |
|---|---|
| `eval_v11/` | `legacy_floorset_arch_eval_floorplans/` |
| `flow_ft_0828/` | `flow_finetune_round1_0828_tailT24_300k/` |
| `flow_ft_0829/` | `flow_finetune_round2_0829_tailT12_250k/`(FT2 = B′ flow prior 來源) |
| `flow_ft_0830/` | `flow_finetune_round3_0830_x0_hpwl_150k/` |
| `flow_ft_0830b/` | `flow_finetune_round4_0830b_bd_cg_mib_140k/` |
| `icdc_g0_v2_area_full/` | `icdc_g0_v2_area_corpus/` |
| `icdc_topology/` | `icdc_topology_prior_training/`(`checkpoints_s2_20k/best.pt` = 出貨 `direct_v2_student_s2.pt`) |
| `icdc_topology_heldout_export/` | `icdc_topology_prior_heldout_export/` |
| `icdc_topology_probe/` | `icdc_topology_prior_probes/` |
| `icdc_topology_train_m16/`、`_m32/` | `icdc_topology_prior_teacher_corpus_m16/`、`_m32/` |
| `logs/` | `training_logs/` |
| `p0_newbox/` | `newbox_gate_runs_0821/`(含 `budget_table_mid.txt`) |
| `shadow/` | `shadow_gate_runs/`(gate chain 的 result JSON / log) |
| (repo root)`shadow_hidden/` | `shadow_hidden_suites/`(v1/v3/v5/v6 shadow 套件、alpha_1、對手面板) |
| (repo root)`checkpoints/` | `legacy_floorset_arch_checkpoints/` |
| `partner/checkpoints/` | `direct_v2_continuation_checkpoints/` |
| (repo root)`contest_optimizer_results.json` | `eval_results/contest_optimizer_results.json` |

docs / scripts 內的 `artifacts/...` 路徑已同步改名;指向早已不存在目錄的歷史引用(`artifacts/partner_eval`、`eval_v10`、`pguide`、`retrieval` 等)保留原文。

## 評測入口(改寫)

`scripts/eval_total.sh` / `eval_single.sh` / `validate.sh` 現在評測**出貨包本身**:預設先用 `pack_cadc1013.sh` 把 `src/` 打到 `artifacts/eval_package/`,再對 `op_wrapper.py` 跑官方 evaluator,結果寫到 `artifacts/eval_runs/`。舊版針對 `src/architecture_v11_optimizer.py` + `FLOORSET_*` env 的邏輯隨 legacy 一併移除;`.env` 縮成 `PARTNER_*` 決策紀錄。
