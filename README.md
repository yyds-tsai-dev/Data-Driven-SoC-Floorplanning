# Data-Driven SoC Floorplanning — ICCAD 2026 CAD Contest Problem C (FloorSet)

隊伍 `cadc1013` 的 SoC floorplanning solver。本 repo 在 2026-09-06 整理為**只保留最終提交路徑**:出貨包的 source code(`src/`)、產生出貨 checkpoint 的訓練工具、打包 / 演練 / gate 工具,以及完整的實驗記錄(`docs/`)。整理對照表:[docs/project-status/2026-09-06-repo-cleanup.md](docs/project-status/2026-09-06-repo-cleanup.md)。

## 最終提交

| 包 | md5 | 內容 | 演練(乾淨 Python 3.13 venv、官方 evaluator、100 案) |
|---|---|---|---|
| **B′** `submission/FINAL_UPLOAD/cadc1013.tar.gz`(08-31 05:20 UTC 上傳) | `3f2cda42c88338fbb32d7670052fc895` | flow prior FT2(round-2 T=12 250k EMA)+ `FLOW_SLOTS 16 / NREF 12` + legal guard 面積修復 + mid 預算表 | **1.0858**、100/100 feasible、avg 0.363 s、max 1.31 s |
| **C′** `submission/FINAL_UPLOAD_C/cadc1013.tar.gz`(08-31 14:47 UTC 綠燈) | `6d0ca94e…` | B′ + FT2×FT3 weight soup α0.5 flow prior | **1.0834**、100/100 |

實際最終上傳者(B′ 或 C′)以隊長確認為準;兩包唯一差異是 flow checkpoint 與 `op_wrapper.py` 一行。工作樹目前是 C′ 設定,`bash scripts/pack_cadc1013.sh <dir>` 重建的包與 C′ 逐檔相同;`bash scripts/release/apply_pack_variant.sh ft2 s16` 切回 B′。beta 階段的 1.314 是部署失敗(空 `requirements.txt` → contest 機缺 scipy、CPU sampler),不是模型品質;決賽演練流程因此加入乾淨 venv 自檢(見 `scripts/release/rehearse_package.sh`)。

## 問題定義(FloorSet Problem C)

輸入一組 SoC blocks(21–120 個):`area_targets`、`b2b_connectivity`、`p2b_connectivity` + `pins_pos`、`constraints`(fixed shape / preplaced / MIB / grouping / boundary)、`target_positions`。輸出每個 block 的 `(x, y, w, h)`。

- **硬限制**:不重疊、soft block 面積誤差 ≤ 1%、fixed-shape 保留尺寸、preplaced 保留位置與尺寸;違反則該案 cost 固定為 M = 10。
- **軟限制**:boundary(貼齊外框邊 / 角)、grouping(同 group 邊相鄰連通)、MIB(同 group 一致 width/height);違反進指數懲罰項。
- **分數**:每案 `cost = quality_factor × violation_factor × runtime_factor`,100 案依 `exp((n − n_max)/12)` 加權,越低越好。大 case 主導總分;runtime factor 是相對全場 median 的 `max(0.7, (rt/median)^0.3)`。
- **`total_score_no_runtime`(noRT)**:本 repo 所有 gate 的主指標(runtime factor 固定 1.0),由 `scripts/iccad2026_evaluate.py`(官方 evaluator 的 repo 副本)輸出。

官方文件、QA 與 alpha / beta 結果在 `docs/official/`。

## Repo 結構

```text
.
├── src/
│   ├── solver/            出貨包的 import closure(26 模組 + contest_optimizer.py + synth_instances.py;檔名 = 包內模組名)
│   ├── shipping/          op_wrapper.py(包內 env setdefaults)、requirements.txt、budget_table_mid.txt、README.md
│   └── icdc_engine/       IC/DC engine + topology-prior trainer(產出出貨的 direct_v2_student_s2.pt)
├── scripts/
│   ├── pack_cadc1013.sh   從 src/ 組出 cadc1013.tar.gz
│   ├── eval_total.sh / eval_single.sh / validate.sh   對出貨包跑官方 evaluator
│   ├── release/           打包變體、乾淨 venv 演練、runtime-aware 分析
│   ├── gate/              五套 shadow gate(run_gate5.sh、run_shadow.sh、analyze_pairs.py、band_pairs.py、ev_rt.py)
│   ├── training/flow_finetune/   出貨 flow prior 的 fine-tune 來源(round 1–4)
│   ├── probes/            仍可執行的實驗探針(package_closure.py 供 pack 使用)
│   ├── build_retrieval_index.py、make_presentation_path_figures.py、install.sh
│   └── iccad2026_evaluate.py   官方 evaluator 副本(加 Total Score (No Runtime))
├── tests/                 pytest(test_solver_*、test_icdc_engine_*、test_postpass_router …)
├── docs/                  全部實驗 / 設計 / 交接紀錄(見下方索引)
├── FloorSet/              官方 submodule(evaluator、dataset loader、LiteTensorData*)
├── submission/            (git-ignored)上傳包、候選包、舊包、包內 checkpoints
├── artifacts/             (git-ignored)fine-tune 輸出、gate 結果、shadow 套件、legacy checkpoints
├── CONTEXT.md、AGENTS.md、CLAUDE.md、.env(PARTNER_* 決策紀錄)、pyproject.toml、uv.lock
```

## 最終 solver(`src/solver/contest_optimizer.py`,包內名 `op_src.py`)

每案 wall-clock 預算由 `PARTNER_BUDGET_TABLE`(n=21..120 各一秒數,mid 表;平均約 0.36 s,n=120 約 1.2 s)決定,流程:

1. **候選供應**(`candidate_supply.py`):
   - **flow prior**(`flow_matching_model.py` / `flow_matching_train.py`,`FLOW_CKPT`):graph-conditioned flow matching,8 步 Euler、antithetic,`FLOW_SLOTS=16` 個候選;出貨權重是 v1(1M 步)的 tail-tilted fine-tune(`scripts/training/flow_finetune/`)。
   - **direct prior**(`direct_diffusion_model.py`、`direct_diffusion_train_v2.py`,`DIRECT_CKPT`):topology-prior 學生模型(DPM++ 2 步),`icdc_engine/` 訓練。
   - 決定性 heuristic seed(pin / graph 加權 centroid)。
   - retrieval 通道(`retrieval_*.py`)為 opt-in,出貨未啟用。
2. **Column-slicing SA legalizer**(`column_sa_legalizer.py`、`sa_numeric_kernel.py`、`csa_coordinate_solver.py`、`column_lns.py`):把候選壓成 column-slicing 佈局,**不重疊 / 面積精確 / fixed & preplaced 保留 by construction**,再以 numba SA 在預算內最小化 HPWL、bbox area、soft violations;24 個 restart worker 平行(`PARTNER_POOL`)。
3. **Refine ladder**(`layout_refiner.py`、`refine_numeric_kernel.py`、`violation_killer.py`、`frame_repack.py`、`tag_compress.py`、`coord_polish.py`、`noise_optimization.py`、`physics_guidance.py`):對模型候選做 edge seat、wall repair、grouping bridge(DAG)、tag compress、frame scale 1.02、final seat;`PARTNER_REFINE_KERNEL=numba`。
4. **選擇與守門**:候選以 evaluator-form cost 排名;`PARTNER_FINAL_LEGAL_GUARD=1` 對最終佈局做 evaluator-faithful 合法性檢查,失敗時先修面積再退到驗證合法的 column 佈局(合法路徑輸出 bit-exact,成本 0.3 ms)。

全部 env 由 `src/shipping/op_wrapper.py` 以 setdefaults 帶入(主辦確認 `op_wrapper.py` 原樣使用);`__init__` 內完成 numba JIT 暖機與 worker pool 啟動(不計時)。每個 knob 的促轉證據見 `docs/experiments/2026-08-21-post-beta-p0-execution.md` §15–18 與 `.env` 註解。

## 安裝

```bash
git clone --recursive <repo-url> && cd Data-Driven-SoC-Floorplanning
bash scripts/install.sh              # submodule、uv、venv、依賴、evaluator smoke test
```

`FloorSet/` 為空時:`git submodule update --init --recursive`。所有 Python 指令一律 `uv run …`。

## 常用指令

```bash
uv run pytest                                   # 全套測試(pyproject: -s, testpaths=tests)
uv run pytest tests/test_solver_final_legal_guard.py -q

bash scripts/pack_cadc1013.sh /path/out         # 組包 -> /path/out/cadc1013.tar.gz(印 entries 數與 md5)
bash scripts/validate.sh                        # 官方 --validate(預設先把 src/ 打到 artifacts/eval_package/)
bash scripts/eval_single.sh 95                  # 單案
bash scripts/eval_total.sh                      # 100 案;結果 artifacts/eval_runs/total_<ts>.json + 摘要行
REPACK=1 bash scripts/eval_total.sh             # 強制重新打包
bash scripts/eval_total.sh submission/FINAL_UPLOAD_C_extracted   # 評測任何已解包的包目錄

bash scripts/release/rehearse_package.sh        # 上傳前演練:打包 -> 解包 -> 乾淨 venv(只裝包內 requirements)-> 官方 evaluator
bash scripts/release/apply_pack_variant.sh ft2 s16   # 把工作樹切成 B′ 設定(無參數 = 0828b)
```

**Gate(促轉鐵律)**:任何 knob 入包前必須「完整候選 env vs 現包 env」同鏈 ×4、兩種臂序,跨 official + shadow v3 / v5 / v6 四套一起看(`scripts/gate/run_gate5.sh`),CI 排除 0 才算;runtime-aware 期望用 `scripts/release/runtime_aware_pairs.py`。方法論在 `docs/project-status/2026-08-28-final-sprint-handoff.md` 與 `docs/experiments/2026-08-21-post-beta-p0-execution.md` §17–18。

**訓練**:
- flow prior fine-tune:`scripts/training/flow_finetune/README.md`(round 1–4 launcher、seed / EMA export 工具);trainer 本體是出貨模組 `src/solver/flow_matching_train.py`。
- topology prior(direct 學生):`uv run python -m icdc_engine.train_topology_prior --help`(在 `src/` 為 PYTHONPATH 下),語料 / checkpoint 在 `artifacts/icdc_topology_prior_*`。

## 文件索引(`docs/`)

- `project-status/`:交接與狀態 — `2026-08-31-final-handoff.md`(最終狀態、B′ / C′)、`2026-08-29-team-summary.md`(給隊友的出貨摘要)、`2026-08-28-final-sprint-handoff.md`(gate 鐵律、否決清單)、`2026-09-06-repo-cleanup.md`(本次整理對照)。
- `experiments/`:逐項證據 — `2026-08-21-post-beta-p0-execution.md`(§1–18w,決賽期間全部 gate)、`flow-finetune/`(fine-tune round 1–3 STATUS)、7–8 月各 gate 記錄。
- `research/`:`2026-08-21-beta-analysis-and-final-sprint-roadmap.md`(beta 分析與 endgame 框架)、論文調查、deep-research 報告。
- `design/`:refine numeric kernel、anytime ladder、early exit、IC/DC engine 設計、slack refiner spec、packaging notes。
- `evaluation/`:5–6 月 legacy 路線的 checkpoint promotion 紀錄(歷史)。
- `official/`:題目、QA、alpha / beta 結果。`presentation/`:期中 / 期末簡報。
- `CONTEXT.md`:術語表(含 legacy 名詞,已標註歷史)。

## 歷史

2026-05 至 07 的 Anchor-GNN / graph-conditioned diffusion / column-backbone 路線(`src/floorset_arch/`,validation noRT 1.24 → 2.1)於 2026-09-06 自 repo 移除(commit `83e8022`,checkpoints 留在 `artifacts/legacy_floorset_arch_checkpoints/`);其設計與證據仍在 `docs/evaluation/`、`docs/experiments/2026-07-*`、`docs/design/slack_refiner_spec.md`。2026-07-12 起改以隊友的 diffusion-seeded column-slicing solver(本 repo 的 `partner/`,現 `src/solver/`)為主線,經 7–8 月的 gate 迭代成為最終提交。
