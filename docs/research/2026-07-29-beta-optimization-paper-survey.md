# Beta 前優化方法論文調研（2026-07-29）

**背景**：ICCAD 2026 FloorSet contest，beta deadline 2026-07-31 17:00（剩 2 天），final 在後。現行 pipeline：GPU diffusion prior（direct_v2, DDIM 25 steps, GPU 段下限 2.0s）→ CPU column-slicing SA legalizer（deadline-bounded）→ slack refine。no-runtime 1.1561 / 投影 ~0.90。官方 per-case Cost=(1+0.5·gaps)·e^(2V)·max(0.7, RTF^0.3)，n=101-120 案佔總分 ~81% 權重。已判死方向（numba SA、<3s cliff、physics guidance、noise-opt、retrieval、CP-SAT）不再重提。

**本報告調研範圍**：A few-step 生成、B GPU/平行 SA、C 快速合法化、D 高 N analytical、E learned compute allocation、F SA move/schedule 學習、G 其他新方法。來源核實等級標注見文末「證據等級聲明」。

---

## 0️⃣ Scheduler 綜合（0729 補記：與內部 0724-0729 實況對齊後的最終結論）

> 以下由主線 scheduler 撰寫。調研正文（第 1 節起）依 0723 快照撰寫，先讀本節的修正再讀正文。

### 內部事實修正

| 正文原述 | 0729 實況 | 影響 |
|---|---|---|
| 檔位一#1「FM few-step probe 待做（0.5 天）」 | **已做完**：flow v1 已訓完，st4/st8/st16 已測（st8 control 1.1443/0.8932 最佳檔位），antithetic 已上（兩 rep 0.8871/0.8855）；目前全場最佳 = **direct_v2@1.2M + flow v1 st8 + antithetic = noRT 1.1385 / proj 0.8859**（`artifacts/partner_eval/direct12M_dfa.json`） | 檔位一#1 收案；st8 已是甜點 |
| 「GPU 段 ~2.0s，few-step 省 >1s/case」 | `PARTNER_DIRECT_MIN=2.0` 是**通道開關閘門**（budget<2.0 的案關閉 direct 通道），非 GPU 耗時。GPU 段實測（`artifacts/flow_matching/v1_final_probe.json`）：DDIM50 0.783s → DDIM25 ~0.39s、flow euler-8 **0.117s**，合計 ~0.5s/case | few-step 直接省時縮水為 ~0.2-0.3s/case；但價值鏈仍在：GPU 段↓ → refine 早開工（DDIM50→25 曾賺 −0.028 noRT）+ DIRECT_MIN 可再降 → 更小 budget 的案也開 direct 通道 |
| 蒸餾以 v1 為基 | v2 已判死（SNR endpoint weighting 定罪，flow worktree `docs/superpowers/plans/2026-07-27-flow-v2-postmortem-and-v2_1.md`）；**v3 mib-hinge 訓至 ~68%，0729 午後完成**；v3@650k 中期 0.8932 持平 control | final 蒸餾對象以 v3 gate 結果為準 |

### 分數結構（優先級框架）

- 投影 0.8859 = noRT 1.1385 × 平均 runtime factor ~0.778；全案吃滿 0.7 折扣 = 0.797 → **runtime 端理論剩餘僅 ~0.09；品質端（尤其 tailQ 1.1292 的 n≥100 案，81% 權重）是大頭**。
- cliff#2@3s = SA/refine 真實地板（numba 判死、colcache wash）→ 3.5s 檔位內品質天花板由「SA 每秒品質」決定 → 能動它的只有 C/D（換合法化引擎）、B（GPU 臂）、legalizer fork 合流。

### 最終行動計畫

**Beta 前（剩 ~2 天）——必做優先，新嘗試讓位：**

0. **[合規·第 0 優先] submission 清理重打包+演練（半天）**：`submission/cadc1013/` 現含 `noise_opt_claude.py`、`physics_guidance_claude.py`、`diffusion_train.py`、`direct_train_claude.py`、`retrieval_*.py` 等多餘 .py——beta spec（`docs/official/beta_test/beta_submission_guidelines_problemC.txt`）明文「多餘/重複 .py、其他嘗試過的做法 = DISQUALIFICATION」。刪除後重打 tar，從 tar 全新解開 → venv → 演練短跑。
1. **[品質] v3 gate（v3 今日午後訓完）**：full-100 A/B vs 0.8859；勝出才換 checkpoint 重打包。
2. **[方差] 定案組合 rep2 複測**：0.8859 為單點，3.5s 檔位 run 方差 ±0.002-0.003，提交前補一 rep。
3. **[選配·時間富餘才做] DPM-Solver++/UniPC 換 direct sampler**（正文 A-1）：25→10 steps，驗 noRT 中性 + `DIRECT_MIN` 降 1.5 可行性；full-100 A/B 不過即棄。
4. **不做**：successive-halving、UCB bandit（工期擠壓必做項，A/B 輪次不夠）、一切訓練類、POOL46 翻案。
5. **提交紀律**：未證項一律不進組態；寧交已證 0.8859。

**Beta 後 → Final（依期望值）：**

1. **FM post-hoc 蒸餾**（arXiv 2510.17858）→ 1-2 step → `DIRECT_MIN`→~1.0、GPU 段 <0.2s（對象 = v3 gate 勝者）。
2. **高 N analytical 原型 gate**（CSF arXiv 2504.03796 最小實作，n>100 20 案子集，贏 column ≥0.01 才加碼）——81% 權重主戰場；DAC'25 LBR 證 diffusion→analytical 同構可行。
3. **確定性合法化**（ICCAD'24 面積遷移）——與 2 共用 constraint-graph 基建。
4. **partner legalizer fork ← src `column_slicing._layout_fast` 合流**（內部項，0723 已標記的大槓桿，正文未覆蓋——partner fork 落後主 repo 版本）。
5. **GPU 空轉臂**（A100 在 CPU SA 期間 ~75% 時間空轉；先過吞吐 microbenchmark gate，colcache wash 前科）。
6. **learned budget 重分配 + move bandit**（排程/策略層，與 1 的省時互為乘數）。

---

## TL;DR 總表

| 方向 | 最佳候選 | 預期收益 | 工程量 | 檔位 |
|---|---|---|---|---|
| A few-step 生成 | ①訓練免費：DDIM→DPM-Solver++/UniPC（25→8-10 steps）②FM v1 直接 few-step Euler probe ③post-hoc shortcut 蒸餾（arXiv 2510.17858，<1 A100-day） | GPU 段 2.0s→0.3-0.8s；解 cliff#2 的直接鑰匙（總時降、CPU SA 時間不變）；runtime 項直接下降 | ①0.5-1 天 ②0.5 天 ③2-4 天 | ①②beta 前 ③final |
| B GPU 平行 SA/packing | A100 空轉期 GPU batch 第二臂（tensorized column eval）；文獻支撐：ISQED'11 GPU-SA 6-160×、PARSAC（IntelLabs 開源） | portfolio 多一臂、零 wall-clock 成本；幅度未知（quality-wash 風險，同 column cache 前例） | 2-3 週 | final |
| C 快速合法化/投影 | ICCAD'24 overlap-graph 最短路面積遷移合法化（10.1145/3676536.3676818）；Per-RMAP feasibility-seeking（TCAD 2025） | n>100 案 SA 前先降 overlap 規模 → SA 每秒更值錢；無 GPU 需求 | 1-2 週（無公開 code，需自實作） | final |
| D 高 N analytical | GrandPlan（ISPD 2026，GPU differentiable FP+placement）；PeF/electrostatics（ICCAD'23）；CSF（arXiv 2504.03796） | n=101-120（81% 權重）換引擎的大注；DAC'25 LBR 證實 diffusion 種子+analytical 精修同構可行 | 2-4 週 | final 大注 |
| E learned budget allocation | Hyperband/successive-halving 用於 parallel restart 早殺重配；per-case marginal-gain 回歸 | 吞吐不變下品質+；不加總時間（符合時間中性鐵律） | restart 早殺 1 天；learned 版 1 週 | 早殺=beta 前可試；learned=final |
| F SA move/schedule 學習 | UCB bandit over move operators（AOS 文獻）；Neural SA（AISTATS 2023，Qualcomm 開源）；CSF 的 Q-learning 步長 | 每步更值錢；bandit 版零模型、零加時 | bandit 1 天+A/B；Neural SA 2-3 週 | bandit=beta 前可試；Neural SA=final |
| G 其他 | MacroDiff+（wirelength-as-representation）；CORE（NeurIPS 2025 RL+EA FP）；DAC'25 diffusion+HGNN+analytical LBR | 主要是路線驗證與 final 期觀察項 | — | 觀察 |

**Beta 前 48h 建議聚焦**：A①（換 sampler）+ A②（FM few-step probe）是唯二「零訓練、只動推理、直接減時」的槓桿；E/F 的 bandit 類是「零加時、純品質」的低風險嘗試，但需 full-100 A/B 承重（「調校即承重」定律）。

---

## A. Few-step 生成（把 25 DDIM steps 壓到 1-8）

**為何是本次最高優先**：GPU prior 段有 2.0s 下限、尾段預算 3.5s；已知 budget <3s 撞 cliff 是因為 CPU 段（refine/SA）有真實地板——few-step 蒸餾把 GPU 段從 ~2.0s 壓到 <0.5s，等於**在不動 CPU 地板的前提下把 per-case 總時間直接砍 1.2-1.7s**，runtime 項（RTF^0.3）直接受益，且不觸犯時間中性鐵律（是減時不是加時）。

### A-1. 訓練免費層（beta 窗口內）

- **DPM-Solver++**（arXiv 2211.01095, Lu et al., NeurIPS 2022 系列）與 **UniPC**（arXiv 2302.04867, Zhao et al., NeurIPS 2023）：高階 ODE solver，**無需重訓**，直接替換 DDIM sampler，一般在 6-10 NFE 達到 DDIM 25-50 steps 品質。repo：`github.com/LuChengTHU/dpm-solver`、`github.com/wl-zhao/UniPC`（皆為知名開源；本節屬背景知識引用，見證據聲明）。
  - **適配點**：`src/floorset_arch/diffusion/sampling.py` 的 DDIM 迴圈換 solver；direct_v2 是座標型 diffusion，低維（≤120×4）、條件強，先驗上比影像更耐 step 壓縮。
  - **收益估計**：25→8 steps ≈ GPU 段 -60~70% 時間。若 GPU 段實測 ~2.0s，可省 ~1.2-1.4s/case。
  - **風險**：樣本分佈微漂移 → legalizer 種子變化 → 終分未必單調（「自洽性>成分品質」前科）。必須 full-100 A/B。**注意**：backbone 的 seed channel 已被測死（GT seed 移動 <0.002），此槓桿的價值主要在 **runtime 減時**而非品質——如果 GPU 段時間不進 per-case 計時則無收益，落地前先確認計時邊界。
- **FM few-step Euler probe**：flow matching 本身軌跡趨直，v1 checkpoint 出爐後直接掃 steps ∈ {1,2,4,8,25} 的品質-時間曲線（`scratchpad/flow_eval_chain.sh` 已有 harness）。零訓練、半天工作量。Rectified flow 理論（arXiv 2209.03003, Liu et al., ICLR 2023；背景引用）預期 FM 在 4-8 steps 的退化遠小於 DDPM/DDIM。

### A-2. 低成本蒸餾層（beta 後、final 前）

- **Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch**（arXiv 2510.17858, Cai et al., 2025-10；abstract 已讀）：**對既有 FM checkpoint 做 post-training velocity-field self-distillation，不需要 shortcut 模型的 step-size embedding、不需重訓**；把 Flux 蒸成 3-step 只花 <1 A100-day，甚至支援 few-shot（10 對樣本）蒸餾。**這是對我們最合身的一篇**：FM v1 checkpoint（h128/l2 級小模型）蒸餾成本估 <1 GPU-hour 量級。repo 未確認（論文未列，需查作者頁）。
- **Shortcut Models**（arXiv 2410.12557, Frans/Hafner/Levine/Abbeel, ICLR 2025 oral；abstract 已讀；repo `github.com/kvfrans/shortcut-models` **已確認存在**，含訓練碼+checkpoints）：單網路單階段訓練，網路額外 condition on step size，任意步數預算可用。**適配點**：FM v2 重訓時直接加 shortcut self-consistency 項（訓練管線改動小）。
- **MeanFlow**（arXiv 2505.13447, Geng/Deng/Bai/Kolter/He, 2025-05；經 websearch 核實）：學 average velocity，1-NFE ImageNet 256 FID 3.43，from-scratch 不需蒸餾。適合 FM v3 從頭訓；官方 repo 未在本次核實。
- **Align Your Flow**（arXiv 2506.14603, Sabour/Fidler/Kreis, NVIDIA；abstract 已讀）：continuous-time flow map distillation，統一 consistency/FM 目標，few-step SOTA 且明確以「小網路」驗證——與我們模型尺寸相性好。
- **域內證據**：**Accelerating Diffusion-based Combinatorial Optimization Solvers by Progressive Distillation**（arXiv 2308.06644, Huang/Sun/Yang；abstract 已讀）——graph-diffusion CO solver（TSP-50）經 progressive distillation **16× 加速、品質僅退化 0.019%**。這是「CO 域 diffusion 可以被激進蒸餾而不掉終分」的最直接文獻證據。Progressive distillation 原始方法見 arXiv 2202.00512（Salimans & Ho, ICLR 2022；abstract 已讀）。
- 其他可參考：SlimFlow（arXiv 2407.12718；小模型 one-step，reflow 初始化不匹配的解法）、MDT-dist（arXiv 2509.04406；25→1-2 steps 的 VM/VD 目標）、InstaFlow（arXiv 2309.06380；reflow 管線）、AnyFlow（arXiv 2605.13724；any-step flow map）。

**方向 A 總結**：beta 前做 A-1（零訓練換 sampler + FM probe），final 前做 A-2（首選 2510.17858 的 post-hoc self-distillation，因為它不要求改 v1 架構）。預期終局：GPU 段 <0.5s，尾段 per-case 總時 3.5+2.0 → 3.5+0.5，全場 runtime 項下探。

---

## B. GPU / 大規模平行 SA 與 packing

- **Optimizing simulated annealing on GPU: a case study with IC floorplanning**（Han/Roy/Chakraborty, ISQED 2011, DOI 10.1109/isqed.2011.5770735；abstract 已讀 via OpenAlex）：對「SA 本質串行」的經典解法——**同一 floorplan 上並行評估大量候選 moves**，MCNC/GSRC 上 6-160× 加速、品質持平或更好。適配點：column-slicing 的 move 評估若能 tensorize（columns × candidate-moves 兩維 batch），A100 可一次評數千 moves。
- **PARSAC**（arXiv 2405.05495, Intel Labs；abstract 已讀；repo **`github.com/IntelLabs/parsac` 已確認**）：massively parallel CPU SA + Constraints-Aware SA（boundary/preplaced 約束由構造滿足）。與我們 column backbone 同族——價值在其 CA-SA move 設計與 Pareto-front 多解管理，可對照撿 move 設計，而非整體替換。
- 通用佐證：CUDA Tabu Search for RCPSP（arXiv 1711.04556；GPU 版同時贏 runtime 與品質）；GPU vertex cover branching（arXiv 2512.18334；不規則搜索樹在 GPU 上可負載平衡）。
- **本 pipeline 特有機會**：**A100 在 CPU SA 期間完全空轉**（prior 只佔前 2s）。GPU batch SA/packing 臂不佔 CPU 吞吐、不加 wall-clock，是「免費 portfolio 臂」。工程路徑：torch 向量化 column cost evaluator（overlap-free by construction 的 column 表示其實適合張量化：每 column 一個 heights vector）→ 數百 restart 並行退火 → 回傳最佳給 CPU refine。
- **風險（誠實面）**：column cache 1.22× 前例證明吞吐↑ ≠ 品質↑（wash）；Python↔GPU 邊界的每步同步開銷可能吃掉收益；工程量 2-3 週起。**判定門檻**：先離線量測「同 wall-clock 下 GPU 臂能否在 n>100 案打平 CPU 48-core 臂」，打不平就殺。

---

## C. 快速合法化 / 投影（SA 之外把 prior 變合法）

- **Modern Fixed-Outline Floorplanning with Rectilinear Soft Modules**（Chen et al., ICCAD 2024, DOI 10.1145/3676536.3676818；abstract 已讀 via OpenAlex；0723 shortlist 項的深化）：其 legalization 是**確定性算法而非 SA**——建 overlap graph 抽取「overlap-module-whitespace 鄰域」，用**最短路把面積從 overlap 沿 module 鏈遷移到 whitespace**，再迭代 expand/shrink 精修 wirelength；rectilinear 形狀自然湧現。fixed-outline+preplaced+soft 全支援，比 SOTA 與 contest 冠軍隊快且好（作者自稱）。**適配點**：對 n>100 案，把 diffusion/FM prior 直接餵此類確定性合法化 → SA 只做 refinement 而非全域合法化，SA 每秒花在更高層的搜索。無公開 code，需自實作（估 1-2 週，overlap graph + Dijkstra + 面積守恆記帳）。
- **Per-RMAP / Floorplanning With I/O Assignment via Feasibility-Seeking and Superiorization**（arXiv 2406.03165 → **TCAD 2025**, vol 44(1):317-330, DOI 10.1109/TCAD.2024.3408106, Yu/Censor/Luo；期刊版 metadata 已核實，arXiv 版 0723 已調研）：把合法化寫成 feasibility-seeking 投影迭代（projections onto constraint sets），superiorization 疊加品質目標。數學上就是我們要的「投影通道」，且天然支持 exact-area 等式約束。
- **CSF**（arXiv 2504.03796v2；abstract 已讀）：宣稱其 CSA（conjugate subgradient）合法化**比 constraint-graph 合法化及其改進變體更高效**——這是「constraint-graph 合法化不一定是最快路」的反向證據，選型時記入。
- **LACE**（arXiv 2402.04754, CVPR 2024；背景引用，已在我方 guidance 基底中用過）：constrained layout 的 gradient post-optimization。guidance 已判死，但其「推理後座標投影」寫法可挪用於 C（在 CPU 端對 prior 輸出先做 5-10 步 overlap 梯度投影再進 legalizer，成本毫秒級）——注意這與 guidance 不同：不動採樣過程，只動 seed。鑑於 seed channel 已測死，此項只在「縮短 SA 到達合法所需時間」上可能有 runtime 價值，優先級低。
- 探索項（低優先）：capacity-constrained power diagram（Balzer et al.；背景引用）可產生 exact-area 凸胞分割，但 rectilinear 化與 soft constraint 對接成本不明。

---

## D. 高 N analytical floorplanning（n=101-120 = 81% 權重）

- **GrandPlan: Differentiable, Simultaneous Top-Level Floorplanning and Partition-Level Cell Placement for Large-Scale IP-Cores**（**ISPD 2026**, DOI 10.1145/3764386.3779591, UT Austin + NVIDIA；經 websearch 摘要核實，ACM 全文 403 未讀）：GPU 端 differentiable 三段式（flat placement with grouping objectives → SA boundary refinement → fence-region placement），**custom CUDA kernels 產生乾淨 rectilinear partition 邊界**，8 顆工業 IP-core（至 25M cells）WL -14%。這是「GPU differentiable floorplanning + rectilinear 邊界」的最新最強存在證明；規模感遠超本題（我們只有 ≤120 blocks），意味此類方法在我們的尺度上會非常快。無公開 code 資訊。
- **PeF: Poisson's Equation Based Large-Scale Fixed-Outline Floorplanning**（arXiv 2210.03293 / TCAD 2023 DOI 10.1109/TCAD.2022.3213609；abstract 已讀；0723 shortlist 深化）：**soft module 的寬度直接作為能量函數變數**參與最佳化 + constraint-graph 合法化。與 exact-area 約束的對接最自然（w·h=A 消去一變數）。無公開 code。
- **Handling Orientation and Aspect Ratio in Electrostatics-Based Fixed-Outline Floorplanning**（Huang et al., ICCAD 2023, DOI 10.1109/ICCAD57390.2023.10323841；0723 shortlist 項，metadata 核實）：ePlace 系電場模型處理 soft module 形狀變數。
- **CSF**（arXiv 2504.03796；abstract 已讀）：nonsmooth 模型 + conjugate subgradient + **Q-learning 自適應步長**，fixed-outline 專用，2025 年新作，MCNC/GSRC 上宣稱勝 SOTA。方法簡單（無需 FFT/Poisson solver），若要 2-4 週內自實作一條 analytical 通道，CSF 路線的實作成本最低。
- **Late Breaking Results: Customized Diffusion Model Empowered by Heterogeneous Graph Network for Effective Floorplanning**（Zheng/Gu/Peng/Wang/Zhu/Zhu, **DAC 2025**, DOI 10.1109/DAC63849.2025.11133070；經 websearch 摘要核實）：HGNN 條件 diffusion 生成初始 floorplan → **餵給 classical analytical floorplanner 精修**（PeF 同組人馬）。**與我們 pipeline 完全同構**（diffusion prior → 傳統 solver），是路線正確性的外部驗證；差異在他們下游是 analytical 而我們是 column SA——支持「n>100 換 analytical 下游」的 final 實驗。
- 周邊：DG-RePlAce（arXiv 2404.13049，GPU analytical placement on OpenROAD，repo 未核實）；AutoDMP（ISPD 2023, DOI 10.1145/3569052.3578923；DREAMPlace + multi-objective 參數自動調優——參數 auto-tune 思路可借給 column SA 的超參）；Escaping Local Optima in Global Placement（arXiv 2402.18311；對 analytical 收斂失敗的 perturb-restart 補丁）；DREAMPlace/Xplace 本體（背景引用：`github.com/limbo018/DREAMPlace`、`github.com/cuhk-eda/Xplace`，皆未含 soft-block floorplanning 模式，需自建能量項）。
- **誠實評估**：此方向是 final 的大注不是 beta 的活。要補的約束（exact-area、MIB、boundary、grouping、rectilinear multi-rect）工程量 2-4 週；且 column 通道「全開飽和 1.2129」說明現引擎還沒到頂，換引擎的邊際要跟繼續調 column 比較。建議 final 期先做 20-case n>100 子集的 CSF-式最小原型再決定加碼。

---

## E. Learned per-case compute allocation / anytime

- **文獻基座**（皆背景引用，本次未重讀）：runtime/品質預測——Hutter et al., *Algorithm runtime prediction: Methods & evaluation*, Artificial Intelligence 206:79-111, 2014；portfolio 選擇——SATzilla（Xu et al., JAIR 32:565-606, 2008）；預算重分配——**Hyperband**（Li et al., arXiv 1603.06560, JMLR 2018）與 successive halving；restart 理論——Luby et al., Inf. Process. Lett. 47:173-180, 1993。
- **新作**：PolarBear（arXiv 2603.08493, 2026-03；abstract 已讀）——anytime 演算法的 Pareto 比較 via Bayesian racing；對我們主要是**離線工具**（比較不同 budget 曲線下的 solver 變體），非在線分配器。
- **適配點 1（beta 前可試，1 天）**：parallel restart SA 目前均分預算；改 **successive-halving**：t/2 時刻砍掉 cost 落後的一半 restart，把核挪給存活者加深搜索。零總時增加、純排程改動、`legalizer/column_backbone.py` 內落地。風險：SA anytime 曲線交叉（後發 restart 反超）導致早殺誤判——先在 20 案 probe 量測 restart 排名在 t/2 與 t 的 Kendall τ，τ>0.6 才上。
- **適配點 2（final）**：per-case marginal-gain 回歸——特徵用既有 instance stats（block count、boundary/group/MIB 密度、net 密度；**禁 test_id**，符合專案鐵律），標籤用歷次 eval 的 per-case (時間,分數) 曲線斜率；在 sum_rt 恆定約束下做 water-filling 重分配。與 0.06·e^(n/20) 公式相比，把「難度≠n」的殘差撿回來。注意 RTF 是 per-case 對全場 median：從 A 方向省出的秒數優先投給「預測邊際收益最高」的案，而非均攤。

---

## F. SA move/schedule 學習（吞吐固定下每步更值錢）

- **Neural Simulated Annealing**（arXiv 2203.02201, Correia/Worrall/Bondesan, AISTATS 2023 PMLR 206:4946-4962；經 websearch 核實；repo **`github.com/Qualcomm-AI-research/neural-simulated-annealing` 已確認**）：把 proposal distribution 當 RL policy 學（小型等變網路），Knapsack/Bin Packing/TSP 上固定算力預算內穩贏手工 SA。**適配點**：column-slicing 的 move（block swap/column transfer/width change）選擇策略可學；但推理成本要 <µs 級才不掉吞吐 → 需 distill 成查表/線性 policy。final 檔。
- **CSF 的 Q-learning 步長調度**（arXiv 2504.03796；abstract 已讀）：population 方案 + Q-learning 調 stepsize，是「輕量 RL 只調 schedule 不換 move」的樣板——遷移成本低於 Neural SA。
- **Reinforcement Learning Based Simulated Annealing**（AAMAS 2025, ACM 10.5555/3709347.3743807；僅 metadata，未讀）。
- **Adaptive Operator Selection with bandits**（Fialho et al. 系列，GECCO/LION 2008-2010；背景引用）+ **Monte Carlo Elites**（arXiv 2104.08781；abstract 已讀，UCB 做 parent selection 的 quality-diversity 實證）：**beta 窗口內最可行的 F 槓桿**——per-restart 對 move 類型跑 UCB1（reward = Δcost/µs），純 CPU 內建、零模型、零加時。工程 1 天 + full-100 A/B。與 partner 的 vkill/carve 邏輯正交。
- **The Impact of Move Schemes on SA**（arXiv 2504.17949；abstract 已讀）：系統性實驗結論「每步動一個隨機粒子最高效」+ partial-coordinate 優於 full-coordinate——對照檢查我們 move set 的粒度設計（column 批量搬移類 move 若存在，值得 A/B 單塊粒度版）。
- **Learning placement order for constructive floorplanning**（Yao/Lin(Yibo)/Li, Integration 2025, DOI 10.1016/j.vlsi.2024.102293；僅 metadata）：學構造順序——對 legacy beam decoder 是舊題（order 品質瓶頸已知），對 column backbone 的 block→column 指派順序可能同樣適用；觀察項。

---

## G. 其他 2024-2026 新方法掃描

- **CORE: Collaborative Optimization with RL and Evolutionary Algorithm for Floorplanning**（Li et al., **NeurIPS 2025**；dblp metadata，未讀 abstract）：RL+EA 混合 floorplanning——「學習體只當 proposal、EA/退火收尾」與我們哲學一致；final 期值得讀全文看 benchmark。
- **MacroDiff / MacroDiff+**（DAC 2025 LBR DOI 10.1109/DAC63849.2025.11132593 + arXiv 2605.16451v2；經 websearch 核實，repo `github.com/jhy00n/MacroDiff` 與 `MacroDiff-plus` 於搜尋結果中存在，未實際開啟——0723 曾記錄 404，狀態以再次點開為準）：**不直接預測座標，改預測 wirelength 關係矩陣作中間表示**，天然旋轉/平移不變，ISPD2005 上 overlap -91.6%、HPWL +7.0%。對 FM v2/v3 的輸出表示是一個真正不同的設計點（座標 frame-blind 問題的另解）。其 Physics-Guided Sampling 部分與我們已判死的 guidance 同類，忽略。
- **DiffPlace**（arXiv 2510.15897；僅 metadata）：conditional diffusion 同時放置——related work。
- **FlexPlanner**（NeurIPS 2024；僅 metadata）：DRL 3D floorplanning——域不合，略。
- **ChipDiffusion**（ICML 2025, arXiv 2407.12282；0723 shortlist 項；repo **`github.com/vint-1/chipdiffusion` 已確認存在**）：其 guided sampling 部分我方已判死；剩餘價值=其 synthetic dataset 生成算法（若 final 期需要擴增訓練分佈多樣性）。
- **DAC 2025 warpage-aware generative floorplanning**（DOI 10.1109/DAC63849.2025.11132923）：先進封裝域，不合。
- 會議雷達：ISPD 2026 accepted list（ispd.cc/ispd2026）除 GrandPlan 外未逐篇掃描；ICCAD 2025/DAC 2026 floorplanning session 未系統掃描（時間盒限制）——final 期補一輪。

---

## 落地建議

### 檔位一：2 天內（beta 前）可落地——原則：零訓練、只動推理/排程、不加時間

按 期望值/風險 排序：

1. **[A] FM v1 few-step probe**（0.5 天，零風險）：checkpoint 一出爐即掃 steps={1,2,4,8,25} 品質曲線。若 4-8 steps 不掉分 → 直接以 few-step FM 作 direct 通道提交 beta，GPU 段省 >1s/case。這是現成整合（b44713a）上的純參數實驗。
2. **[A] DDIM→DPM-Solver++/UniPC**（0.5-1 天，中風險）：diffusion 通道 25→8-10 steps。風險=種子分佈漂移過 legalizer 後的終分非單調；**full-100 A/B 是唯一裁判**。若 FM 通道 probe 已勝出，此項可跳過（replace-not-add）。
3. **[E] restart successive-halving**（1 天，中風險）：先量 restart 排名穩定性（Kendall τ probe），過門檻才上 full-100 A/B。不改總時、不加核。
4. **[F] move-type UCB bandit**（1 天，中風險，與 3 擇一先做）：per-restart UCB1 選 move 類型，reward=Δcost/時間。與 partner 現行 move 權重靜態表相比是嚴格泛化。
5. **beta 提交紀律**：以上任何一項未在 full-100 上證明 ≥中性（品質）且 runtime 尾端無回歸，就不進 beta 提交組態——寧交已證的 1.1561/0.9008。

### 檔位二：beta 後 → final 路線

按 期望值 排序：

1. **[A] FM post-hoc shortcut 蒸餾**（arXiv 2510.17858 路線，2-4 天）：v1 → 1-2 step。模型極小，蒸餾成本估 GPU-hour 級。若成，GPU 段下限 2.0s→<0.3s，配合 [E] 把省下的秒數重分配到高邊際案。備選：Align Your Flow / shortcut objective 進 v2 重訓 / MeanFlow 進 v3。
2. **[D] 高 N analytical 最小原型**（1 週 gate 實驗 → 過門檻再 2-3 週）：CSF 式 subgradient 或 PeF 式 Poisson，先只做 n>100 的 20 案子集、只當 portfolio 臂與 column 對打。DAC 2025 LBR 證明 diffusion 種子 + analytical 精修同構可行。**Gate：子集上贏 column ≥0.01 才繼續。**
3. **[C] 確定性合法化通道**（1-2 週）：ICCAD 2024 最短路面積遷移自實作，目標=n>100 案把「到達首個合法解」的時間從 SA 手裡拿走，SA 全時做品質。與 2 可共用 constraint-graph 基建。
4. **[B] GPU 空轉期第二臂**（2-3 週，成敗五五開）：tensorized column eval + 數百並行退火。先做吞吐 microbenchmark gate（GPU 臂單位時間 cost 下降速度 ≥ CPU 48 核臂），不過就殺，避免重蹈 column cache wash。
5. **[E] learned per-case 預算重分配**（1 週）：以歷史 eval 曲線訓 marginal-gain 回歸，water-filling 分配。與 1 的省時互為乘數。
6. **[F] Neural SA / 學習型 proposal**（2-3 週，最後上）：吞吐與表示穩定後才值得；否則策略學到的是舊 move 空間。
7. **[G] 觀察項**：CORE（NeurIPS 2025）與 MacroDiff+ 的 wirelength 中間表示——若 v3 重訓立項，先讀全文再定。

### 跨方向約束備忘

- 任何品質宣稱以 **full-100 no-runtime A/B** 承重；任何 runtime 宣稱附 raw runtime 尾端分佈（RTF 是 per-case 對全場 median，尾段案權重 e^(n/12) 支配）。
- 時間中性鐵律 0723 修訂後：「減時」是一等槓桿，「加時」仍禁（oracle run 除外）。
- 所有觸發條件用 instance 統計，禁 test_id。

---

## 證據等級聲明（誠實標注）

- **本次 session 已讀 abstract（工具回傳全文摘要）**：2510.17858、2506.14603、2410.12557、2308.06644、2202.00512、2407.12718、2509.04406、2309.06380、2605.13724、2504.03796、2210.03293、2402.18311、2404.13049、2405.05495、2504.17949、2104.08781、2603.08493、2605.16451、ISQED 2011（via OpenAlex）、ICCAD 2024 3676536.3676818（via OpenAlex）、AutoDMP（via OpenAlex）。
- **經 websearch 二手摘要核實（未讀原文）**：MeanFlow 2505.13447、GrandPlan（ISPD 2026，ACM 全文 403）、DAC 2025 LBR 11133070、MacroDiff/MacroDiff+、Neural SA 2203.02201（PMLR 頁面確認）。
- **僅 metadata（dblp/搜尋列表）**：CORE（NeurIPS 2025）、Per-RMAP TCAD 2025 期刊版（arXiv 版 0723 已調研）、Yao/Lin/Li Integration 2025、AAMAS 2025 RL-SA、FlexPlanner、DiffPlace、ICCAD 2023 electrostatics（0723 shortlist 已核）。
- **背景知識引用（本次未驗證，屬既有共識文獻）**：DPM-Solver++ 2211.01095、UniPC 2302.04867、rectified flow 2209.03003、consistency models 2303.01469、Hyperband 1603.06560、Hutter AIJ 2014、SATzilla JAIR 2008、Luby 1993、Fialho AOS、Balzer power diagrams、DREAMPlace/Xplace repo 座標。
- **Repo 存在性本次核實**：`kvfrans/shortcut-models` ✔、`IntelLabs/parsac` ✔、`Qualcomm-AI-research/neural-simulated-annealing` ✔、`vint-1/chipdiffusion` ✔、`jhy00n/MacroDiff(-plus)`（搜尋結果顯示存在，未開頁驗證）。PeF/CSF/Per-RMAP/GrandPlan/ICCAD'24 **無已知公開 code**。
- 預期收益數字（如「GPU 段 -60%」「<1 A100-day」）除引文自帶者外均為**推估**，未經本 repo 實測；一切以 full-100 A/B 為準。
