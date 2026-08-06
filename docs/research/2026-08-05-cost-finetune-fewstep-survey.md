# Cost/Reward Fine-Tuning 與 Few-Step Legality-Aware 生成調研（2026-08-05）

**任務**：回答兩個決策問題 —
Q1「cost/reward fine-tuning 路線是否成熟可移植？」、
Q2「2025-2026 有沒有能在 ~0.05-0.1s/case 給出接近合法 floorplan 的 few-step、legality-aware 生成工作？」

**操作點**：每案平均 0.2s（tail 0.44s），noRT 1.308 → 目標 1.000。
**不重複調研**：`docs/research/2026-07-29-beta-optimization-paper-survey.md`（蒸餾 / analytical / SA move / anytime）與 memory `paper-methods-shortlist-0723`。本篇只補 reward-FT 與 legality-aware few-step 兩塊。

---

## 1. TL;DR 決策建議

1. **Q1 = 條件式，預設「不投資」。** 方法本身**確實成熟且可移植**（Flow-GRPO 有官方 repo 且我方 flow v1 正是它的目標模型類；非可微 reward 有 DDPO/SEPO policy-gradient 與 LaSRO 可微 surrogate 兩條成熟路），成本對我們是 **GPU-hour 級**（模型 h128/l2，瓶頸在 CPU rollout 不在 GPU）。
2. **但擋路的不是演算法，是通道。** reward-FT ≡ **noise-opt 的離線攤提版**：兩者都在同一個「種子 → backbone → refine」通道上，用同一個標量目標壓生成分佈。我方 0723 已實跑 noise-opt 全 100：**no_runtime 1.1539 → 1.2714（+0.1175），n≥100 帶 +0.120、20/21 案變差**；敗因是「同質化候選擠掉尾段 SA/refine 依賴的結構多樣性」。reward-FT 在這條軸上**只會更糟**（它塌縮的是整個先驗分佈，不是單案種子）。
3. **唯一能繞開的變體**：把 reward 定義在 **best-of-N 組合層**（BoN-aware / vBoN），而非單樣本。GRPO 天然是 group-relative，只要把 group 的 **min-cost** 當 reward 就近似免費拿到這個性質。
4. **若投資，第一步不是訓練，是量測**（半天、零 GPU）：固定 portfolio N，量 `d(total_score_no_runtime)/d(種子品質)` 的**分 band 符號**，特別是 n≥100 帶；evaluator 的 seed 方差 σ≈±0.0008 已知，量測靈敏度綽綽有餘。**尾段符號非負才解鎖訓練**，否則直接結案。
5. **Q2 = 此路不通（誠實結論）。** 沒有任何 2025-2026 工作同時滿足「NFE≤4 + 非事後 clip 的 non-overlap 機制 + 有 repo」。最接近的 FlowPlace（arXiv 2604.23658, 2026-04）是 **20-50 Euler steps、無 repo、固定尺寸 macro**（我們是 soft block + exact-area + MIB）。
6. **Q2 唯一可撿的碎片**：FlowPlace 的 **x̂₁-投影修正速度**（extrapolate → project → corrected velocity），零額外 NFE 就換到 by-construction 合法。但**對我們現值 ≈ 0**（refine 本能修 raw overlap、種子通道已測死），只在通道活性被證實後才值得掛上 sampler。
7. **真正對「golden 1.05 imitation 天花板」有效的文獻**是 data-free cost 訓練（IC/DC 2411.00003、Sanokowski 2406.01661/2502.08696）——但那是**換引擎的 final 大注**，不是 beta 槓桿；且我方 realize 品質 2.35 vs bar 0.3 的舊證據說明距離仍遠。
8. **一句話**：imitation 天花板不是當前的 binding constraint，「下游不消費先驗」才是。Q1 修的是非約束項，Q2 修的是已被 refine 免費提供的東西。

**若仍要投資，具體配方（base = flow_matching v1, euler-8）**：Flow-GRPO（2505.05470，repo `yifan123/flow_grpo`）的 ODE→SDE + Denoising Reduction（訓練 4 步、推理 8 步）；每個 instance 抽 N=8，**reward = 該 group 經 backbone+refine 後的最佳終分（非 raw 指標）**，GRPO group-relative advantage；KL 係數必須小（大 KL 會把增益吃光，小 KL 會塌縮多樣性——這正是 gate 要先量的東西）。

---

## 2. Q1 候選表：cost / reward fine-tuning

| # | 方法 / 論文 | arXiv | Repo | reward 型態 | 成本量級 | 移植風險 / 對我們的判讀 |
|---|---|---|---|---|---|---|
| 1 | **Flow-GRPO**（Training Flow Matching Models via Online RL, NeurIPS 2025） | 2505.05470 | `github.com/yifan123/flow_grpo`（搜尋結果確認為官方實作，未實際開啟） | 任意黑箱標量（group-relative，無需可微） | 原論文在 SD3.5-M 上多卡數天；**我方模型小 3-4 個數量級，GPU 可忽略，成本全在 CPU rollout** | **最合身的 base**。兩個關鍵設計正中我們痛點：ODE→SDE 轉換讓 deterministic flow 可做 RL 探索；**Denoising Reduction**（訓練少步、推理原步數）直接壓訓練成本。風險 = 通道死（見 §4） |
| 2 | **DDPO**（Training Diffusion Models with RL） | 2305.13301 | 專案頁 `rl-diffusion.github.io`（repo URL 未在 abs 頁確認） | 非可微、黑箱 | 中（多步 policy gradient，reward query 量大） | 經典基線。把去噪當多步 MDP 做 policy gradient；**與我們「reward = 下游 solver」設定相容**，但 reward query = 一次完整求解，query 效率是主成本 |
| 3 | **DRaFT / DRaFT-K / DRaFT-LV** | 2309.17400 | abs 頁未列 code | **需可微** reward | 低-中（DRaFT-K 只回傳最後 K 步） | 我們的 reward 含合法化管線 → **不可微，直接不適用**，除非走 #4 的 surrogate |
| 4 | **LaSRO**（Reward FT of ≤2-step DMs via learned latent surrogate reward, CVPR 2025） | 2411.15247 | 專案頁 `sites.google.com/view/lasro`（repo 未確認） | 把**任意（含不可微）reward 學成可微 surrogate**，再走梯度 | 中（多訓一個 reward model） | **概念上最貼我們的處境**：明確論證 PPO/DPO 在 ≤2-step 蒸餾模型上失效，改用 surrogate。缺點是**我們的 reward 是「經過 SA+refine 之後」的分數，surrogate 要學的是 solver 的行為**——這等價於重建一個可微 solver proxy，難度不低於本體 |
| 5 | **SEPO**（Score Entropy Policy Optimization，離散 diffusion） | 2502.01384 | `github.com/ozekri/SEPO`（abstract 明列） | 明確支援**非可微** reward | 低-中 | 若我們改成離散表示（order / column 指派 / sequence pair）才有意義。目前架構無離散生成器 → 備用 |
| 6 | **IC/DC**（Unsupervised diffusion for feasible solution generation in NCO） | 2411.00003 | 未確認 | **直接把 cost + 約束當訓練目標，無需 expert label** | 從零訓練 | **唯一正面攻擊 golden-1.05 imitation 天花板的路**。且「不需 problem-specific search 就能產生 valid 解」＝ by-construction 合法。但這是**換引擎**，非 fine-tune；final 大注 |
| 7 | **Sanokowski 等**（rKL-bound data-free diffusion for CO；Scalable Discrete Diffusion Samplers, ICML 2024 / ICLR 2025） | 2406.01661, 2502.08696 | 未確認 | 目標函數本身（unnormalized 分佈） | 從零訓練 | 同 #6 的理論基座（policy-gradient / SN-NIS 兩種記憶體友善訓法）。列為 #6 的方法論支撐 |
| 8 | **FTDiff**（GRPO-style RL-FT + fast sampling，分子生成，2026-05） | 2606.01220 | 未確認 | 多目標 threshold-aware 標量 | 未報告 | **最新的「RL-FT ＋ few-step 同時做」示範**（結構約束下、免 post-hoc 處理）。價值 = 證明兩者可疊，不是可直接移植的碼 |
| 9 | **DiOpt**（Self-supervised diffusion for constrained optimization, ICML 2026） | 2502.10330 | 專案頁（abstract 提及） | 目標 + 約束（自監督 bootstrapping） | 中 | 兩階段：supervised warm-start → bootstrapping 自我精進。**與我們「先 imitation 再 cost-FT」的自然路徑同構**，可當工程樣板 |
| 10 | **vBoN / Inference-aware BoN FT** | 2407.06057, 2412.15287 | 未確認（**僅搜尋摘要，未開 abstract**） | BoN 分佈（保留多樣性） | 低（是 objective 改寫，不是新管線） | **解鎖 Q1 的關鍵補丁**：直接對 best-of-N 的分佈做對齊，而非壓單樣本。我們的 portfolio 正是 BoN，這是唯一與「多樣性塌縮」失敗機制不衝突的目標函數 |
| 11 | **TRS**（Trust-Region Noise Search，黑箱推理期對齊，2026-03） | 2603.14504 | 「source code publicly available」（URL 未捕獲） | 完全黑箱（生成器 + reward 都當黑箱） | 零訓練，**但每次生成要多次 reward 評估** | **對 0.2s 操作點 DoA**：每次 reward eval = 一次完整求解。且這正是 0723 已判死的 noise-opt 家族（ReNO/D-Flow/DNO）的黑箱版 |
| 12 | 綜述：Alignment of Diffusion Models（ACM CSUR 2026） | 2409.07253 | `github.com/xie-lab-ml/awesome-alignment-of-diffusion-models` | — | — | **僅搜尋摘要**。若真的開案，用它當方法地圖，不必自己再掃 |
| 13 | 多樣性保全：Perceptual Entropy for flow-based RLHF | 2605.12112 | 未確認（**僅搜尋摘要**） | — | — | 標題即承認「policy entropy 約束會失效」——側面佐證多樣性塌縮是 flow RL-FT 的已知病 |

**Q1 判準逐項回答**：
- **(a) 有無可跑 repo**：有。Flow-GRPO（官方，且針對 flow matching）與 SEPO（離散）是兩個確定存在的落地點。
- **(b) reward 不可微時的方案**：兩條成熟路 —— policy-gradient（DDPO/Flow-GRPO/SEPO，**我們該走這條**）或學一個可微 surrogate（LaSRO）。我們的 reward 含合法化管線 → surrogate 路等於要學 solver，不划算。
- **(c) 訓練成本量級**：對我們**不是 GPU 問題**。模型 h128/l2，梯度步的 GPU 成本可忽略；成本 = rollout。粗估：N=8 draws × batch 32 instances × 每 rollout ≈0.2s CPU ≈ 51 s/step 單核 → 1000 steps ≈ 14 CPU-hours 單核 / 48 核約 0.3 小時純算。**真實代價是它與 evaluator 搶同一台 48 核機器**，不是 GPU-hours。（此為估算，未實測。）
- **(d) 「reward 由下游 solver 決定」有無成功先例**：**沒找到 EDA 域的先例**。三次不同角度搜尋（placement + RL-FT + legalizer/HPWL reward、GRPO + macro placement、warm-start + solver-in-the-loop）都沒有命中。最接近的是 #6/#7 的 data-free NCO（但那是 cost 直接可算、無下游 solver）與 #9 DiOpt。**這是一個沒有先例的設定，風險應相應計入。**

---

## 3. Q2 候選表：few-step、legality-aware layout 生成

判準：有 checkpoint/repo、NFE≤4、有處理 non-overlap 的**機制**（非事後 clip）。

| # | 方法 / 論文 | arXiv | Repo | NFE | non-overlap 機制 | 判讀 |
|---|---|---|---|---|---|---|
| 1 | **FlowPlace**（Flow Matching for Chip Placement, 2026-04/06） | 2604.23658 | **無**（全文未提供） | **20-50 Euler steps**（非 ≤4） | **有，且是好機制**：每步 extrapolate `x̃₁ = xₜ+(1−t)v_θ` → **hard constraint projection** `x̂₁=C(x̃₁)`（GPU 平行 greedy legalization，作者稱 negligible overhead）→ 修正速度 `(x̂₁−xₜ)/(1−t)` | **Q2 最接近的一篇，但三條不合**：步數不達標、無 repo、**macro 是固定尺寸矩形**（我們是 soft block + exact area + MIB 矩形線性化）。可撿的是 x̂₁-投影這個 pattern（見 §4） |
| 2 | **HardFlow**（Hard-Constrained Sampling for Flow-Matching via Trajectory Optimization, 2025-11） | 2511.08425 | abs 頁未確認 | 不降；**額外加 MPC/最佳控制** | 在**終端時刻**精確滿足硬約束（明確批評 per-step 投影「過度限制、傷品質」） | 方向對（終端約束 > 全程投影），但**加算力不減算力**，與 0.05-0.1s 預算反向 |
| 3 | **Projected Diffusion Models** | 2402.03559 | 未確認（**僅搜尋摘要**） | 不降 | 每步投影到約束集 | 家族基座；同樣加算力 |
| 4 | **Predict-Project-Renoise (PPR)**（2026-01） | 2601.21033 | 未確認（**僅搜尋摘要**） | 不降 | predict→project→renoise 迭代 | 同上 |
| 5 | **Adaptive Correction Scheduling**（2026-05） | 2605.11214 | 未確認（**僅搜尋摘要**） | 不降 | 早期鬆、後期緊的漸進約束 | 排程技巧，可搭 #1 |
| 6 | **Chance-constrained Flow Matching**（2025-09） | 2509.25157 | 未確認（**僅搜尋摘要**） | 不降 | 機率型約束 | 我們要的是硬約束，錯配 |
| 7 | 離散/一步：Coupling Models for One-Step Discrete Generation | 2605.07193 | 未確認（**僅搜尋摘要**） | **1** | 無 layout 專屬約束機制 | 一步離散生成的通用機器；若未來走「生成 sequence pair / order」才相關 |
| 8 | **ChipDiffusion**（ICML 2025）/ **DiffPlace** | 2407.12282, 2510.15897 | `github.com/vint-1/chipdiffusion` 已知存在 | 1000 步 | guided sampling（我方 guidance 已判死） | 0723/0729 已調研，不重複 |

**Q2 結論：沒有一篇同時過三個判準。** 「NFE≤4 + 由構造保證 non-overlap + 有 repo」在 2025-2026 文獻中**不存在**（至少本次三輪搜尋未命中）。

且即使存在，對我們的期望值仍接近 0：**refine 本能修掉 raw overlap**（F4 guidance 判死的機制）、**種子通道已測死**（GT 座標種子動總分 <0.002）。要讓 Q2 有價值，生成器必須是**取代 backbone 的終端輸出**，不是種子——而在 0.05-0.1s 內端到端贏過 0.2s column SA 的證據，我方（scale probe realize 2.3499 vs bar 0.3）與文獻（FlowPlace 需 20-50 步且是固定尺寸 macro）都不支持。

**唯一保留項**：FlowPlace 的 x̂₁-投影修正速度。它比我方判死的 physics guidance 形式更好（作用在 x₁-預測上而非梯度推擠、零額外 NFE、by-construction），但**現值 ≈ 0**，僅在通道活性被重新證實後才掛上。

---

## 4. 與已判死路線的區隔（為何這不是 F4 guidance / flow v2-v3 的重演）

**先說結論：Q1/Q2 都無法自動宣稱區隔，反而必須先承認同構。**

| 已判死路線 | 機制 | reward-FT 是否同構 |
|---|---|---|
| **F4 physics guidance**（採樣期梯度推擠合法性） | refine 本能修 raw overlap → guidance 功勞被抹平 | **同構**。任何以「讓生成物更合法/更低 raw cost」為目標的介入，功勞都會被 refine 吸收 |
| **noise-opt（ReNO/D-Flow/DNO 族，0723）** | 凍權重、對 input noise 做能量梯度下降。Phase A 全綠（能量降 28%、同算力 raw overlap 贏 baseline_big），**full-100 大敗 +0.1175**；n≥100 +0.120、20/21 案變差 | **高度同構**。reward-FT = 把 noise-opt 的 per-case 搜索**攤提進權重**。通道相同、失敗機制相同 |
| **flow v2 / v2.1 / v3、diffusion fv2、direct_v2 續訓** | 五個訓練賭注全部邊際或判死 | reward-FT 是第六個訓練賭注；沒有理由預設它會不同，除非**目標函數的形狀真的不同** |

**真正的區隔只有一個，而且必須被明確設計進去：目標函數的形狀。**

- 之前所有介入（guidance、noise-opt、imitation 訓練）優化的都是 **單樣本的 raw 品質**。
- noise-opt 的敗因報告寫得很清楚：「對 seed 做 overlap 最小化產生**同質化**的近合法候選 …… 擠掉尾段 SA/refine 依賴的**結構多樣性**」。
- 這正是 RL-FT 文獻裡有名字的病：**reward over-optimization / mode collapse**（DPOK、Flow-GRPO 用 KL 罰項擋，但 KL 大就沒增益、KL 小就塌縮）。**單樣本 reward-FT 會系統性地放大這個病**，因為它塌縮的是整個先驗分佈，而不只是單一案例的種子。
- **唯一結構性不同的目標**是 §2 的 #10：把 reward 定義在 **best-of-N 的分佈** 上（vBoN 2407.06057 / inference-aware BoN 2412.15287）。GRPO 的 group-relative advantage 只要把 group 的 **min-cost（而非每個樣本各自的 cost）** 當訊號，就近似免費得到 BoN-aware 性質——這時「多樣性」不是要被罰的副作用，而是被獎勵的資產。

**因此解鎖條件（Gate，做完才准動 GPU，成本 <0.5 天、零訓練）**：

1. **通道活性 + 符號量測**：固定 portfolio N 與所有下游設定，只擾動生成先驗的品質（可用既有 flow v1 的 temperature / 不同 checkpoint / GT 種子三檔），量 full-100 **分 band**（n<60 / 60-100 / **n≥100**）的 `total_score_no_runtime`。evaluator seed 方差 σ≈±0.0008，靈敏度足夠。
   **通過條件：n≥100 帶的斜率符號非負。**（noise-opt 在此為 +0.120 = 明確為負。）
2. **BoN 敏感度**：同一組先驗下，比較 `E[cost of single draw]` 與 `E[min over N draws]` 對先驗品質的斜率是否**異號**。若異號，就證明了「單樣本目標會傷、BoN 目標才會幫」，直接指定 objective。
3. 兩關都過 → 才跑 Flow-GRPO 配方（§1 第 8 點）。任一關不過 → **結案，寫進判死清單**，把預算還給 §2 的 #6（data-free cost 訓練，final 大注）或 0729 survey 已排定的蒸餾/analytical 路線。

---

## 5. 參考文獻（arXiv id）

**Q1 — reward / cost fine-tuning**
- 2505.05470 — Flow-GRPO: Training Flow Matching Models via Online RL（NeurIPS 2025；repo `yifan123/flow_grpo`）
- 2305.13301 — Training Diffusion Models with Reinforcement Learning（DDPO, Black et al.）
- 2309.17400 — Directly Fine-Tuning Diffusion Models on Differentiable Rewards（DRaFT, Clark et al.）
- 2411.15247 — Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward（LaSRO, CVPR 2025）
- 2502.01384 — Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods（SEPO；repo `ozekri/SEPO`）
- 2411.00003 — IC/DC: Unsupervised Training of Diffusion Models for Feasible Solution Generation in NCO
- 2406.01661 — A Diffusion Model Framework for Unsupervised Neural Combinatorial Optimization（ICML 2024）
- 2502.08696 — Scalable Discrete Diffusion Samplers（ICLR 2025）
- 2606.01220 — FTDiff: Fine-Tuning Diffusion Models for Molecular Generation via RL and Fast Sampling
- 2502.10330 — DiOpt: Self-supervised Diffusion for Constrained Optimization（ICML 2026）
- 2407.06057 — Variational Best-of-N Alignment
- 2412.15287 — Inference-Aware Fine-Tuning for Best-of-N Sampling
- 2603.14504 — Trust-Region Noise Search for Black-Box Alignment of Diffusion and Flow Models
- 2409.07253 — Alignment of Diffusion Models: Fundamentals, Challenges, and Future（ACM CSUR 2026；清單 repo `xie-lab-ml/awesome-alignment-of-diffusion-models`）
- 2605.12112 — When Policy Entropy Constraint Fails: Preserving Diversity in Flow-based RLHF
- 2607.14522 — A Continuous-Time RL Framework for Fine-Tuning Discrete Diffusion Models

**Q2 — few-step / legality-aware layout 生成**
- 2604.23658 — FlowPlace: Flow Matching for Chip Placement
- 2511.08425 — HardFlow: Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization
- 2402.03559 — Constrained Synthesis with Projected Diffusion Models
- 2601.21033 — Predict-Project-Renoise: Sampling Diffusion Models under Hard Constraints
- 2605.11214 — Enforcing Constraints in Generative Sampling via Adaptive Correction Scheduling
- 2509.25157 — Chance-constrained Flow Matching for High-Fidelity Constraint-aware Generation
- 2605.07193 — Coupling Models for One-Step Discrete Generation
- 2407.12282 — ChipDiffusion（ICML 2025，0723 已調研）
- 2510.15897 — DiffPlace（0729 已調研）
- 2510.23472 — BBOPlace-Bench: Benchmarking Black-Box Optimization for Chip Placement（背景）

---

## 6. 證據等級聲明（誠實標注）

- **本次實際開啟 abstract 頁並讀取內容**：2604.23658（含 HTML 全文 method 節）、2502.01384、2406.01661、2606.01220、2411.15247、2502.08696、2511.08425、2411.00003、2502.10330、2305.13301、2309.17400、2603.14504。
- **僅由搜尋結果摘要獲得（未開 abstract）**：2505.05470（但摘要含足夠的 method 細節：ODE→SDE、Denoising Reduction、NeurIPS 2025、官方 repo）、2402.03559、2601.21033、2605.11214、2509.25157、2605.07193、2407.06057、2412.15287、2409.07253、2605.12112、2607.14522、2510.23472。
- **Repo 狀態**：`ozekri/SEPO` 由 abstract 明列；`yifan123/flow_grpo`、`xie-lab-ml/awesome-alignment-of-diffusion-models`、`vint-1/chipdiffusion` 由搜尋結果確認存在但**未實際開啟**；FlowPlace **確認無 repo**（全文未提）；其餘標「未確認」。
- **工具限制**：`paper-search-mcp` 的 arXiv 後端本次全程回傳空結果（arXiv 對本機 IP 回 HTTP 429），改以 WebSearch + `arxiv.org/abs` 直取；因此**本篇的召回率低於一次正常的 arXiv 全庫掃描**，「找不到符合 Q2 三判準的工作」是三輪不同 query 的結果，不是窮盡證明。
- **未驗證項**：§2 的 rollout 成本估算（14 CPU-hours 級）為紙上推算，**未實測**；本次依指示**未跑任何本機評測**。
- **內部證據引用**：noise-opt 全 100 數字取自 `docs/experiments/2026-07-23-runtime-frontier-ddim25-dm2.md`「補記 2」；seed 方差 σ≈±0.0008 同出處。
