# 2026-08-21 Beta 結果分析與最終衝刺路線圖

**輸入**:beta 官方結果(`docs/official/beta_test/`)、對方組前端報告(`docs/research/前端報告.md`)、我方後端報告、學長「多產 training data」建議、gpt 5.6-sol 半完成研究(`.planning/` + TFDL 系列文件)、新一輪 paper search(12 篇)。
**方法**:evidence 彙整(3 個 fact-gathering agents)→ 定量推演 → deep-reasoner(Opus)對抗覆核 → 修正後合成。本文所有分數口徑:官方 `exp(blocks/12)` 加權、no-runtime = RuntimeFactor 固定 1。

---

## 0. TL;DR

1. **我們 beta 第 6,但這名次是借來的。** 我們 total/raw = 0.9245/1.3207 = **0.70004,100 案全部貼死 runtime floor 0.7**;而 rank 1/5/8 還沒貼 floor。若決賽人人貼 floor(理性終局),**名次由 raw quality 單獨決定,我們掉到第 8**。
2. **Runtime 已到瓶頸——方向反了。** 再降 runtime 期望收益 = 0(僅剩保險價值);正確玩法是**反向把時間花回去**:floor 允許每案花到 0.3046×median,我們只用了 52s/90s。本地可再花 1.7–2.3× 時間、零罰分。**但要買的是 quality,不是 runtime 本身**——照我們實測曲線,時間加倍只值 −0.010~−0.013,所以加時間是「解鎖器」,不是主菜。
3. **主菜是 violation 消滅。** 本地 weighted V_rel=0.027,乘數 exp(2v)=1.056 → 清零值 **−0.060 raw**,比其他任何單一 lever 都大,而且是正確性修復、最能轉移到 hidden。
4. **local→hidden 落差 +0.156(1.1647→1.3207)比所有 lever 都值錢。** 最能解釋全部落差的假說:**hidden 上 soft violation 變高**(exp(2Δv)=1.155 恰好吻合 1.3207/1.1437 比值)。機器速度假說量化上死亡(≈0.011,且本地同為 48 核)。判別實驗一天內可做(見 §4)。
5. **QA A12 確認:final test = beta 同一組 hidden testcases。** beta median CSV 就是決賽題目的中位數表(決賽會重算 median,但題目不換)——per-case 預算重校直接成立。
6. **5.6-sol 的 TFDL topology prior:G0-v2 已過(教師預測性增益 7.11× 目標),G1 因果實驗未跑。** 建議以背景 GPU 工作收尾(P(pass)≈30–40%,期望 −0.015~−0.03),A/B 移到 pseudo-hidden 集上做。
7. **學長「多產 data」建議的正確落地:FloorSet 沒有 generator;data 已經存在**——`floorset_lite` 9,000 shards 的 heldout(TFDL 已索引 21,081 筆 n≥100)。最高價值用法是**建 300–1000 案 pseudo-hidden 評測集**(反過擬合 + 落差診斷),其次才是模型訓練語料。
8. **合規紅線:** beta 報告要求 final `requirements.txt` **必須完整列出所有依賴**、torch>=2.5.0(Python 3.13)。我們 0810/0812 包是 0-byte requirements(Case A)。**final 前必須改成完整版並在乾淨 Python 3.13 venv 實測**——這是 disqualification 級風險,成本半天。

---

## 1. Beta 結果精讀

### 1.1 計分與 floor

```
cost_i = (1 + 0.5·(max(0,hpwl_gap)+max(0,area_gap))) · exp(2·V_rel) · rt_adj_i
rt_adj_i = max(0.7, (t_i / median_i)^0.3)      # median_i = 該案所有隊伍 runtime 中位數
total = Σ w_i·cost_i / Σ w_i,  w_i = exp(blocks_i/12)   # top-10 案 = 56.6% 權重,top-20 = 81%
infeasible = 10.0;  feasible cost 下限 = 1.0(gap 有 max(0,·) 夾住)
```

- rt_adj 貼 0.7 的條件:`t_i ≤ 0.3046 × median_i`。
- **rank 1 raw = 1.0027,距理論下限 1.0 僅 0.27%**——他們在 hidden 上幾乎每案打平或超越 golden(HPWL 負 gap 被夾零)。該隊 raw 已無空間,只剩把 runtime 壓到 floor(0.8236→0.7019)。

### 1.2 Leaderboard 兩種讀法

| beta 名次 | total | raw | runtime | rt 比值 | 貼 floor? | 全 floor 化 total | **raw 名次** |
|---|---|---|---|---|---|---|---|
| 1 | 0.8258 | 1.0027 | 169.4s | 0.824 | 否 | 0.7019 | **1** |
| 8 | 0.9424 | 1.1242 | 83.1s | 0.838 | 否 | 0.7870 | **2** |
| 2 | 0.8537 | 1.2077 | 110.7s | 0.707 | ≈是 | 0.8454 | **3** |
| 5 | 0.9081 | 1.2211 | 95.3s | 0.744 | 否 | 0.8548 | **4** |
| 3 | 0.8932 | 1.2556 | 60.3s | 0.711 | ≈是 | 0.8789 | **5** |
| 4 | 0.8993 | 1.2848 | 24.5s | 0.700 | 是 | 0.8993 | **6** |
| 7 | 0.9252 | 1.3141 | 38.2s | 0.704 | ≈是 | 0.9199 | **7** |
| **6(我們)** | **0.9245** | **1.3207** | **52.1s** | **0.70004** | **是** | **0.9245** | **8** |
| 9 | 1.0322 | 1.4643 | 81.7s | 0.705 | ≈是 | 1.0250 | 9 |
| 10 | 1.0598 | 1.5140 | 14.1s | 0.700 | 是 | 1.0598 | 10 |

已有 4–5 隊貼 floor;還沒貼的恰是 quality 最強的幾隊(r1/r8/r5),他們把 runtime 壓下來是零成本動作。**規劃基準必須用 raw 賽局**:要贏 r7 需 −0.007、r4 −0.036、r3 −0.065、r5 −0.100、r2 −0.113、r8 −0.197、r1 −0.318(以我們 beta raw 1.3207 為基準)。

### 1.3 我們的提交是哪一版、落差多大

- Beta 送的是 **0810 resubmission 包**(md5 `c2dd42bb…`;`repack_checklist_0729.md` §8):本地 official100 **1.1647**(RF=1)、avg 0.293s。
- Hidden raw **1.3207** → **落差 +0.156(+13.4%)**。
- 現役最新包 **0812 groupbridge**:本地 **1.1437**(−0.021 vs beta 包)→ 樸素外推 hidden ≈1.300 → floored total ≈0.910,仍在 rank 5/6 邊界(r5=0.9081)。**光靠現有進度追不上 raw 賽局的前四。**

---

## 2. Runtime:瓶頸判定與正確操作(直接回答)

**問:runtime 是否已到瓶頸?繼續降是否會讓 score 無法再降?**

**答:是,已飽和。** 100 案全貼 0.7 floor(§1.2),繼續降 runtime 的分數收益**恰為零**;若再為降時間犧牲任何 quality(SA 步數、候選修復時間),分數只會變差。5.6-sol「reduce-time / anytime ladder」路線追求 avg 0.2s 的那個目標**應正式作廢**(其中 refine-kernel 的加速是 quality-positive 的,已促轉、保留)。

反向操作的定量框架(以 beta median CSV,QA A12 確認 final 同題):

| n(blocks) | median 擬合* | floor 點(0.3046×) | κ=0.20 預算 | 本地換算(÷1.78) |
|---|---|---|---|---|
| 60 | 2.05s | 0.62s | 0.41s | 0.23s |
| 100 | 4.46s | 1.36s | 0.89s | 0.50s |
| 110 | 5.42s | 1.65s | 1.08s | 0.61s |
| 120 | 6.58s(實測 9.9) | 2.00s | 1.32s | 0.74s |

\* `median(n) ≈ 0.639·e^(n/51.5)`(log-resid σ=0.154);case 99 實測 9.9s 高於擬合 +50%,尾端以擬合值(保守)為準。總量:floor 點總預算 90.1s(contest 機),我們 beta 用 52.07s;本地換算可花 **1.7–2.3× 現在的時間**。

執行守則(deep-reasoner 覆核後修正):

1. **先重分配、後加總量**:同總時間下把小案(權重 <0.1%)預算搬到 top-20 權重帶,零風險,先吃掉大部分價值。
2. **κ 取 0.18–0.20,不要 0.25**:final median 會漂移(整個 field 讀了 beta 報告都會動)。κ=0.25 時「越 floor 期望罰分」≈「quality 期望增益」互相抵消;κ≈0.2 以下期望罰分 <0.2%。且 rt_adj 冪次淺(rf=0.5 也只 ×0.812),超一點不是災難——**infeasible 才是災難**。
3. **災難尾管理是鐵律**:最重案 infeasible 一案 = total +0.87 ≈ 直接墊底。任何加時間/加重修復的變更必須帶:硬 wall-clock deadline + 既有 row fallback + 記憶體防護(128GB 上限,QA A3)。**寧慢勿死**:超時到 10×median 只 ×2.0,infeasible 是 ×8。
4. 預算 key 用 block_count(可重用統計量,符合「不得 hard-code test_id」方針),`PARTNER_BUDGET_{SCALE,TAU,MIN,MAX}` 都是 env 旋鈕,重校免改碼。

---

## 3. 優化路線圖(打折後期望值;≈14 天到 final)

deep-reasoner 修正:各 lever **不可直接相加**(機制重疊 + official100 有效樣本 ESS≈24 的 winner's curse),整包打 ~5 折估。目標設定:本地 1.1437 → **1.08–1.10(現實)/1.05(拉伸)**;若落差診斷能收回 0.03–0.08,hidden raw 可進 1.17–1.24 → raw 賽局第 3–5。

### P0(本週,全做)

| # | 項目 | 期望(打折) | 成本 | 說明 |
|---|---|---|---|---|
| A | **final 合規修包**:完整 `requirements.txt`(torch>=2.5、scipy、numba 全列)+ 乾淨 **Python 3.13** venv 全流程實測 + `getpass`/`TORCHINDUCTOR_CACHE_DIR` 容器防護 | 避免 DQ | 0.5 天 | beta 報告 §2a/§4 checklist 明文;我們現行 0-byte Case A 在 beta 能跑不代表 final 安全 |
| B | **Violation 消滅計畫**:P2(frame 只 pin 高不 pin 寬 → R-tag preplaced 60.5% 違規)、R2(`_stack_column` 同 cluster 相鄰 by-construction)、殘餘 boundary/grouping 外科 pass 強化 | **−0.03~−0.05** | 3–4 天 | V_rel 0.027→~0.005 理論值 −0.060;修復類最能轉移到 hidden |
| C | **預算重校 + 解鎖架上品**:§2 守則落地;重開「有品質但當時超時」項——selective LP joint repair(n≥100,實測 −0.0109)、R1-SOCP full(0.68s/case 現在買得起)、P4 column 寬度收窄重試 | −0.02~−0.03 | 2–3 天 | 對**現行** baseline 重測增量,不能沿用歷史數字(R1-cheap 已入產線) |
| D | **落差診斷 + 反過擬合**(§4):perturbation test、heldout-500 pseudo-hidden、同包 5× 重跑測 σ、(視結果)把 32-config/閾值在 pseudo-hidden 上重調 | 落差收回 0~−0.08 | 2–3 天 | **期望值最高的單項**;學長建議的落地 |

### P1(與 P0 並行/其後)

- **TFDL G1 收尾**(5.6-sol 半成品;Task 5 學生訓練 + Task 6 凍結 A/B):攻的正是 HPWL/topology 瓶頸(weighted HPWL gap 0.114 = 最大 quality 項)。教師 G0-v2 預測性增益 ΔH=0.186(7.11× 目標、21,081 筆 heldout n≥100)。但學生要過兩層損耗(學到教師 + 過合法化保留),模型側 prior 在本 repo 的轉移前科 0/4 → **P(pass)≈30–40%**。守則:背景 GPU 跑、不佔關鍵路徑;**G1 的 −0.015 判準移到 pseudo-hidden 上驗**(ESS≈24 的 official100 上 −0.015 在雜訊帶內);訓練學生時採對方組實證的 train-longer(200k+ steps、workers=4、檢查 dataloader-bound)。
- **選解 proxy 校準稽核**:IEEE Access 2026 敵隊論文(§7 #1)實測「raw-magnitude 排序 vs gap-based 官方口徑失準值 ~2%」——對照稽核我們 evaluator-form proxy 與 1.5% 切換閾值。0.5 天。
- **每個 lever 一律 paired per-case delta + bootstrap CI**(ESS≈24,單跑 σ≈0.008–0.010;沒有 CI 的 −0.01 都當雜訊)。

### P2(有餘裕才做)

- Re²MaP 式 **group-level 搬遷 move**(整個 MIB/cluster 跨 column 移動,新 SA move class)。
- LNS destroy 選擇用 proxy 分數導向(現有 WINDOW_REPACK 的 scored 版)。
- Fast-SA 三段溫度排程(同預算更深收斂;「調校即承重」風險,先 tail-5 paired)。
- Order-channel probe(OrderPlace):**先對照 0707 kill 證據**(golden-order 餵入曾大幅變差)——只有「SA 內建構順序」與「目標順序模仿」被證實不同機制才投入。

### 不做(雙方實測互證的死路)

前端/模型 raw 品質投資(對方組 96.7% 利用率前端 + 弱後端 = 1.216,比我們差;兩隊獨立測得「raw 更好 ≠ 最終更好」)、physics guidance(+0.0070)、noise-opt(+0.1175)、更多候選(+0.0706)、座標 seed 通道(<0.002)、GPU SA、ILP/CP-SAT、懶人 AOT 編譯。

---

## 4. +0.156 落差:假說與判別實驗

| 假說 | 量化 | 判別實驗(全部本地、便宜) |
|---|---|---|
| **(iv) hidden 上 soft violation 升高**(外科 pass 是照 official100 樣態調的,guard 不過就回退) | exp(2Δv)=1.155 **恰可解釋全部**(Δv≈0.072) | **perturbation test**:對 official100 做題目級擾動(面積、連線權重、constraint 抖動)re-solve,看 V_rel 是否跳升。1 天 |
| (iii) 操作點過擬合 official100(預算曲線、32-config、閾值;ESS≈24) | 0.02–0.05 | heldout-500(floorset_lite)評測:若 heldout ≈ official100 → 沒過擬合;若差 → 在 heldout 重調。1–2 天 |
| (i) 機器較慢 → SA 步數變少 | ≈0.011(2× 時間才值 −0.011;且本地同 48 核) | 已可判定為次要;免做 |
| (ii) hidden 部分案較難(官方自述 "occasionally harder") | 殘差項 | 由上兩實驗的殘差反推 |
| 抽樣運氣 | σ(total)≈0.05 → +0.156 ≈ 3σ,**大概率是真的** | 同包 5× 重跑先把 run-to-run σ 釘住 |

**結論導向**:若 (iv) 成立,P0-B(violation 消滅)就是整場比賽;若 (iii) 成立,P0-D 的 pseudo-hidden 重調直接回收 0.02–0.05。兩者都比任何新算法便宜。

---

## 5. 5.6-sol 工作盤點(`.planning/` + progress)

| 工作 | 狀態 | 處置建議 |
|---|---|---|
| flow-matching v1(`checkpoints/flow_matching_v1/final.pt`,LFS 已入庫) | **已入產線**(3-Flow slots,0729 起) | 不動 |
| reduce-time / anytime ladder | ladder 全檔位 wash **判死**;refine-kernel 突破(−0.009~−0.011,bit-exact)**已促轉** | 路線目標(avg 0.2s)作廢(§2);kernel 保留 |
| groupbridge(PARTNER_TAG_COMPRESS + GROUP_BRIDGE) | **現役最新包** 1.1437 | final 基底 |
| **TFDL data-free topology prior** | Task 1–4 完成、G0-v2 **TARGET_GAIN_MET(7.11×)**;Task 5(學生訓練)、Task 6(G1 凍結 A/B)未跑 | **P1 收尾**,判準移 pseudo-hidden(§3) |
| server migration | 本地 main 已整併(2,260 passed);**GitHub push 因憑證卡住**;17 個 topology scorer-import 測試失敗待 re-baseline(evaluator 從 submodule 而非 `scripts/` 解析) | 工程債:補憑證推送(含 15GB LFS);修 scorer import 路徑後再宣稱全綠 |

## 6. 對方組前端報告:拿什麼、不拿什麼

- **戰略結論(雙方互證)**:他們 58.6M 擴散前端 raw 品質極高(area gap 0.002、util 96.7%)但合法化把 util 打到 78–86%、最終 1.216;我們合法化面積損耗已壓到 0.049、最終 1.1437。**接縫(合法化保留拓撲)才是勝負,不是前端**——兩隊各自實測都指向同一句話,我們的「不投前端」決策再獲外部確認。
- **可拿**:①「train longer ≫ scale up」(12k→200k steps 值 −0.14,容量提升統計不顯著)+ dataloader-bound 診斷法 → 用在 TFDL 學生訓練;②他們 15-step>25/50/100 的少步取樣結論與我們 DPM++ 2-step 一致(互證);③他們的失敗清單(overlap loss 無效、放大模型無效、overlap-as-predictor 失效)= 我們 kill list 的獨立重演,省掉重試成本。
- **不拿**:x̂₀ 硬投影(我們 08-05 調研已判價值≈0)、其表示法(我們已有等價的 aspect 解析式)。

## 7. 學長建議(多產 training data)的落地

FloorSet **沒有** instance generator(07-07 調研 + repo 實查確認)。「更多 data」的三個可行形態,按價值排序:

1. **Pseudo-hidden 評測集**(P0-D):從 `floorset_lite` heldout 抽 300–1000 案(fp_sol/metrics_sol 齊,TFDL 已建 21,081 筆 n≥100 索引)。用途:promotion gate 的統計功效(官方 100 案 ESS≈24 太小)、落差診斷、操作點重調。**成本最低、直接攻 +0.156。**
2. **TFDL 學生訓練語料**(Task 5 既定計畫,receipt-bound training-split)。
3. (選)teacher-labeled 擴充:對訓練 instance 做擾動、用 TFDL 教師產標籤(教師需 fp_sol 種子,擾動題無 fp_sol,需契約內變體設計)——僅在 1、2 之後考慮。

## 8. Paper search 新增(12 篇,已排除既知清單)

| 排序 | 論文 | 用途 |
|---|---|---|
| 1 | **Gap-Aware Candidate Reranking on FloorSet**(IEEE Access 2026, DOI 10.1109/ACCESS.2026.3722685) | 唯一已發表的外部 FloorSet 基線(比我們差:Lite 1.402);**借:metric-calibration 稽核法(~2%)**、兩態候選結構與 DiT 放大負結果(又一次互證) |
| 2 | OrderPlace(ICML 2026, arXiv 2606.08904) | 建構順序 = 未開發維度;離線蒸餾零 runtime 成本;**先過 0707 order-kill 對照** |
| 3 | Re²MaP(arXiv 2511.08054) | packing-tree group 搬遷 → column 上的 group-level move class(P2) |
| 4 | LNS + neural destroy(CPAIOR 2026, arXiv 2603.20801) | proxy 導向「拆哪段 column」(P2) |
| 5 | TTA for CO(TMLR 2026, arXiv 2601.21048) | public→hidden shift 的模型側參考;CPU 預算存疑,僅模型 arm 存活才看 |
| — | 其餘:DAC25/TCAD26 hetero-graph diffusion(平行發明,skim)、UPC flow-matching thesis、inference-time scaling ×2(與已死 noise-opt 家族重疊)、beam-search reduction、WISP whitespace(Lite 適用性低)。**Dry hole(誠實負結果)**:restart portfolio bandit、slicing-tree 精確 DP 2024–2026 無新貨 | |

## 9. 名次推演(誠實版)

- 全 lever 打折落地:本地 1.1437 → **1.08–1.10**;落差若收回一半(0.08):hidden raw ≈ **1.16–1.19** → 全 floor 化 total 0.81–0.83 → **raw 賽局第 3 上下**。
- **Rank 1(1.0027)在現有 local→hidden 映射下不可達**(我們的架構天花板 repaired-golden 1.0113 外推 hidden ≈1.15–1.17)——除非落差被大幅消滅。所以**落差工作 = 名次工作**。
- 對手也在動:r4(0.245s/case,已貼 floor)有 3.7× 時間可花;r8 修掉 runtime 就是 0.787。把 §1.2 的 floored 表當作**對手模型 artifact** 持續更新,提交決策以它為準。

## 附錄:本輪未及、留給下一棒

- 同包 5× 重跑 σ 實測(所有 CI 的分母)。
- 官方未公布 per-case 成績/我們的 per-case contest runtime;若 final 前有 QA 窗口,問:per-case 上限(觀測 max 2930s 未被殺)、final median 重算口徑。
- beta 報告 §2 其他隊踩的坑(torch ABI、getpass、缺檔)全數對照自查一次(P0-A 內)。
