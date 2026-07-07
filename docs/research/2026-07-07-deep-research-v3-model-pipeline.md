# 2026-07-07 深度研究 v3:模型架構 × 管線 × 訓練方法(合流報告)

**目標**:降 `total_score_no_runtime`(production 基準 **1.2297**,GT 直評 1.1079,golden-order exact decoder 上限 **1.0535**)。
**方法**:deep-research workflow(5 角度 / 22 一手來源 / 110 claims 抽取 / 25 claims 進 3 票對抗驗證:**23 confirmed、2 refuted、0 unverified**;104 agents)→ 兩位獨立審查者(deep-reasoner/Opus 讀碼審查 + Codex 讀碼審查)對抗檢驗 scheduler 的候選排序 → 本文合流。
**與上輪差異**:上輪(`.claude/plans/2026-07-07-order-decoder-roadmap.md`)查證階段因 usage limit 全滅(25 unverified);本輪全部 claims 完成 3 票驗證,並補上模型/訓練角度。上輪排的 Track-A 槓桿(D1/A/B/N1/N2/N3)已全數量測判死——本輪聚焦唯一存活的路線 C 與其周邊。

---

## 0. TL;DR(合流後結論)

1. **文獻高度收斂於本 repo 的實測診斷**:瓶頸在「學 block ordering / pairwise 相對位置」,不在座標預測。保留 column-slicing SA legalizer 當精確 packer,把模型改造成上游 ordering/pairwise 訊號源——這是文獻與本地證據共同指向的架構。
2. **但兩位審查者一致指出排序裡最薄弱的假設:「learned order 會是 golden-like」完全未經驗證**,且 0706 的 pair-head-v2 判死(標籤=幾何規則套 golden 質心,模仿標籤打不贏幾何抽取)已給出近乎決定性的反證機制。**任何 Route C 訓練投資之前,先跑零訓練成本的 golden-oracle 上限 probe(Gate 0)**——零件已全部存在於 `gen_decoder_probe_v2.py`。
3. **訓練監督必須換標籤**:LaMPlace(ICLR 2025 oral)證明 pairwise 係數可用「solver 自產 layouts + 下游精確評分 + ranking loss」離線蒸餾(僅 ~1,200 筆);本 contest 的 evaluator 便宜,此路徑完全可行,且正是 0706 kill 註記要求的方向。Vlastelica blackbox diff 降為第二階段。
4. **管線分工(回答本輪核心問題)**:文獻(FlowPlace/ChipDiffusion,3-0 驗證)支持「graph encoder + coarse 生成 + inference-time guidance + exact repair」,constraints 不進 denoiser;**本 repo 的特化修正:座標介面已三重判死,有效介面是 ORDER**——「coarse 生成」的正確輸出是 pairwise/order 結構(或只經 order 抽取消費的座標),group/MIB/boundary 交給 realizer 的 constraint pass + production refine 疊層(D1 屍檢:decoder 自帶 constraint pass 遠弱於 refine 疊層,是 violation 稅的直接來源)。
5. **反面證據(全 3-0 驗證)**:RL 座標 placement 打不贏 SA(Kahng v3:9 案 SA 勝 6、算力 21×);proxy 在高品質前沿鑑別力崩壞(Kendall τ 0.402/0.051)——「HPWL 強 surrogate 為 V10 proxy 背書」的說法被 0-3 否決,V10 前沿鑑別力是未驗證的開放風險(但 production path 根本不走 v10 proxy,此風險只影響 Route C 的候選排序,對策=直接用 exact evaluator 排候選)。
6. **FloorSet submodule 沒有資料產生器**(雙審查者讀碼證實:只有 HuggingFace download + loader)——「自產 constraint-clean 樣本」是新專案不是槓桿,降級。

---

## 1. 文獻 findings(全部經 3 票對抗驗證)

| # | 內容 | 來源 | 票 |
|---|---|---|---|
| F1 | **learned ordering > learned placement**:RL 只選「下一個放哪個 block」,位置交確定性程序(枚舉 SP 插入點擇優);vs learned baseline WL −25.2%(MCNC;單一 baseline、無獨立複現) | Yao/Lin/Li, Integration VLSI 2024, 10.1016/j.vlsi.2024.102293 | 3-0×3 |
| F2 | **LaMPlace**:GNN 輸出 position-agnostic per-pair Laurent 係數,一個 instance 只算一次;離線監督僅 ~1,200 筆 placement-label 對,MSE + weighted pairwise **ranking loss**;可插入任何 sequential/BBO placer。code: MIRALab-USTC/AI4EDA-LaMPlace | ICLR 2025 oral | 3-0/2-1/3-0 |
| F3 | **O-tree/CBL + pointer network 選序**:TODAES'24 勝 tuned SP-SA 10.58% WL / 3.88% area,**但 runtime 3.3×**;KDD'22 RL 直接預測 block id+CBL 位置。⚠ 被否決:「CBL 解碼 O(n) legality by construction」(0-3) | 10.1145/3653453;10.1145/3534678.3539220 | 3-0×3 |
| F4 | **edge-aware attention**:edge embedding 投影為 attention bias + node/edge 逐層聯合更新(作者未單獨 ablate edge-awareness;機制同 Graphormer 系) | 同 TODAES'24 | 3-0 |
| F5 | **FlowPlace**:flow matching + 每步「外推→greedy 合法化投影→corrected velocity」,構造性零重疊,zero-shot 秒級 ICCAD'15 rWL avg rank 1.62;**ChipDiffusion**:training-free backward universal guidance,zero-shot legality 0.8835→0.997、HPWL −7.1%,quality/legality 全在 denoiser 外 | arXiv 2604.23658(DAC'26);arXiv 2407.12282(ICML'25)+ vint-1/chipdiffusion | 3-0×4 |
| F6 | **合成資料+domain priors+curriculum**:ChipDiffusion 全用程序化合成資料 zero-shot 遷移;FlowPlace 證明結構性 prior(boundary-bias 取樣)勝隨機合成;**泛化斷崖**:edge-length scale>0.4、超過訓練 vertex 數 → legality 驟降(支持 scale probe + n 覆蓋檢查) | 同上 | 3-0×4 |
| F7 | **ChiPFormer**:offline RL(decision transformer)純離線學 place-one-block policy;移植需把 fp_sol 分解為軌跡+外加 order(循環依賴 ordering)→ 只宜作 order 標籤基建 | arXiv 2306.14744(ICML'23) | 3-0 |
| F8 | **反面**:Kahng v3——SA 在 RL 自家 proxy 上 9 案勝 6,CPU-hours 19,068 vs 896;proxy 前沿 Kendall τ 崩壞(0.402/0.051);ChiPBench——六 SOTA AI placer 中間指標贏、端到端 PPA 全輸。⚠ 被否決:「HPWL 強 surrogate 為 V10 背書」(0-3) | arXiv 2302.11014v3(2026-03);arXiv 2407.15026 | 3-0×4 |

---

## 2. 雙審查交叉結果

### 一致點(兩位獨立得出,信心高)

1. **「learned order 是 golden-like」是最薄弱假設**,必須先量上限再投資。0706 已證:pair-head-v2 的標籤 = 把 `build_order_dags` 的幾何規則套在 golden 質心上(`losses.py:205-253` docstring 明寫 "matching EXACTLY"),模仿此標籤的模型打不贏幾何抽取本身(0.858 vs 0.897);那 1.3% 難 pair 是 near-tie、**標籤本身接近隨機**。
2. **Vlastelica blackbox diff 不該首發**——先做靜態 evaluator-蒸餾 head(掛現有零件),過 gate 再上結構化 solver 求導。
3. **edge-aware attention 降級為 ablation**:(a) 標籤噪音不是特徵表達力問題;(b) Codex 讀碼:現有 HGT/GraphTransformer encoder 已有 edge/relation bias(`nn/model.py:10-65,99-212`),「補 edge-aware」不是明顯缺口;(c) 動 encoder 有 anchor 漂移前科(0706)。
4. **V10 τ 校準不是獨立槓桿**:production path 在 `optimizer.py:176` 短路,完全繞過 v10_proxy;τ 崩壞風險只影響 Route C 候選排序 → 對策併入 gate:**Route C 的 order 候選排序直接用 exact evaluator(本地免費),不用 V10 proxy**。
5. **FloorSet 無產生器**(`FloorSet/lite_dataset.py:24` 只有 HF download;全 submodule 搜 generator 零命中)→ 合成資料自產降級為長線,近期資料側只剩 scale/curriculum/clean-sample 過濾(`--clean-sample-policy strict` 已存在)。
6. FlowPlace 式 in-sampler projection 維持停車,等 scale probe(0709)判定 diffusion 是否復活。

### 分歧點與裁決

| 議題 | deep-reasoner | Codex | 裁決(scheduler) |
|---|---|---|---|
| Gate 0 的預測 | 預測 golden-oracle probe **失敗**(order 救不了 production 幾何) | 未預測,但要求同 probe 量 head/geom/comp agreement + tail 分層 | 不預判,跑了才算;兩人的 probe 設計合併(見 §3 Gate 0) |
| 更高 EV 新槓桿 | 提出 E1(tail 表示切換)+ E2(案內預算重配) | 「沒有比 Route C 更高 EV 的新干預」,補 tail-weighted golden-likeness harness | E2 收(便宜、正交);E1 收但**修正 pre-check 設計**(見下);tail harness 併入所有 gate |
| E1 pre-check 用 GT 座標過 refine | 建議 | — | **修正**:GT 直評 1.1079 已勝 production,GT-過-refine 有 strict-better guard 兜底,結果 trivially 勝——是 oracle 無資訊。改用 **realizable 生成器**(offline 大預算 SP-SA / partner diffusion 座標)對 tail-5 per-case 比較才有資訊量 |

---

## 3. 合流後排序表

**時序錨點**:scale probe 正式判定 = ep5-6(~07-09,預註冊規則:清楚逼近/跨越 OLD → 存活)。

| 優先 | 干預 | 具體內容 | 時序 | kill / promotion 判準 |
|---|---|---|---|---|
| **Gate 0** | **golden-oracle order 上限 probe**(新,零訓練) | `gen_decoder_probe_v2.py --pair-blend`,pair_map 接現成 `_golden_pair_class`(v2:89-97)。兩變體:(a) production hints——「order 對了能否救 production 幾何」;(b) diffusion hints(scale ckpt)——Route C 的**可實現上限**。同 run 記 head/geom/comp agreement、fallback、area_gap/fill、full-100,**tail(n≥116)分層報表** | 立即,0709 前 | (a)+(b) 都 ≥1.228 → **ML-order 全線降級**,資源轉 E2/E1;(b) 顯著 <1.2297 → Route C 上限確立,給出 head 需達到的 agreement 門檻 |
| **1** | **scale probe ep5-6 判定**(既定)+ n 覆蓋檢查 | 既定儀器與規則;附加(F6 斷崖):確認訓練分布覆蓋 n→120 上界,不足則 curriculum 補 | 0709 | 預註冊:未清楚逼近 OLD → diffusion 座標路線死,ML 只剩 Gate 0 通過後的 Route C |
| **2** | **Route C v1:evaluator-蒸餾 pair-cost head(LaMPlace-lite)** | **凍結 encoder/anchor**(0706 鐵律);現有 v2 pair head 換監督:**solver 自產 layouts(SA restart 產物/擾動)+ exact evaluator 分數 + weighted pairwise ranking loss**——學「什麼 order 得分高」,不是「golden 長什麼樣」,繞開標籤循環;consumer = 現成 `build_order_dags_blend`(v2:100-146,高信心才 override) | Gate 0 通過後,與 scale 並行 | roadmap §5 gate 補 tail 分層:head-beats-geom>0(**特別在 n≥116**)、area_gap<+0.3、fill>85%、full-100 >0.005 |
| **3** | Route C v2:Vlastelica blackbox diff | 結構化 order solver(argmin w·φ,雙軸 acyclic)+ Hamming loss + perturbed backward(λ≈10-20) | #2 停滯後的備選,不並行 | 同 #2;pair 決策已被 DAG longest-path 投影解耦,其邊際價值存疑(deep-reasoner 推論) |
| **4** | **Route C 兌現端強化:decoder 輸出接 production refine 疊層** | D1 死因=decoder 自帶 constraint pass 弱於 refine 疊層(vsnap v1+v2+violation-aware cost)→ 把 decoder 輸出(合法者)過 `refine/`(pure function+guards,表示無關),砍 violation 稅;與 Track B v2 的 repair_pin_order/aspect repair 銜接 | 與 #2 並行(工程小) | golden-oracle/模型 order 經此管線的 v_rel 稅不降 → 不採 |
| **5** | FlowPlace 式 flow matching + in-sampler projection | diffusion 復活後的 guidance 升級:取代現行 geometry-gradient guidance(sampling.py:197-238);projection operator 用 column legalizer 的廉價近似(seed+單 restart);constraints 留 projection/repair 層 | scale 存活且 Track B v2 bar 通過後 | 每步 projection 的 runtime 過不了提交風險評估 → 退回 post-hoc |
| **6** | **E2:tail 案內時間預算重配 probe**(便宜、正交) | `_time_budget`(column_backbone.py:28-32)對 n≥116 的 24s clamp 放寬(×2),paired 量測;合法(案內加時,非已判 INVALID 的跨案中位數謬誤) | 隨時,不依賴 ML | tail-5 paired <0.003 → kill;**RTF 提交期風險必須另行標記**(慢懲罰不封頂,恰在最重案) |
| **7** | E1:tail 表示切換 pre-check(探索) | 修正版:**realizable** 非 column 生成器(offline 大預算 SP-SA、partner diffusion 座標)對 tail-5 per-case 比 HPWL;動機:column 表示 HPWL 天花板 ~1.35×GT,tail 5 案佔加權 37%(test99 獨佔 10.45%) | idle-time | 全敗 → tail 天花板是硬的,回 Route C;注意 A-kill 機制(violation 稅)仍在,兌現需 #4 |
| ablation | edge-aware attention | 只作 #2 訓練時的 encoder ablation | — | — |
| 併入 | V10 τ 校準 | Route C 候選排序直接用 exact evaluator | — | — |

### 0707 量測後記(本表發布數小時後,Gate 0 與 E2 已實跑;證據 [docs/experiments/2026-07-07-gate0-order-channel-e2-tail-budget.md](../experiments/2026-07-07-gate0-order-channel-e2-tail-budget.md))

- **Gate 0 ☠ 觸發**:(a) production×golden-order **9.2090**(0勝100敗,fallback 79/100);(b) diffusion×golden-order **9.3663** vs self-order 同 ckpt 對照 **8.9496**——golden order 反而更差。**order-only 通道判死**;機制=headroom 需要 (order, shape) 聯合構型,order 單獨攜帶資訊 ≈ 0。
- **級聯**:#2 Route C v1(LaMPlace-lite)與 #3(Vlastelica)的 order-override consumer 形式**連帶判死**;#4(兌現端強化)失去 order 客戶、降回停車;**ML 唯一存活路徑 = #1 scale probe(座標+形狀品質,0709 判定),order 由幾何抽取免費取得**。
- **#6 E2 ✓ 存活(今日唯一)**:tail n≥116 配對 **5/5 全勝、加權 −0.00441**(bar 0.003),v_rel 零變化、全量持平(噪音對照對稱);促轉前置=第二組確認配對 + RTF 提交期風險評估。旗標休眠落地(`FLOORSET_TAIL_BUDGET_SCALE/_N`)。
- **0707 晚二輪量測**(證據 [docs/experiments/2026-07-07-channel-matrix-e2-bands-bestofn.md](../experiments/2026-07-07-channel-matrix-e2-bands-bestofn.md)):形狀通道也 ☠(PG 8.98/DG 8.77),**通道不可分解矩陣完成——自洽性>成分品質,oracle 拼裝全滅**;E2 **N=100 擴帶 −0.0128(total 1.2161 歷史最佳)**,劑量反應 scale1.5 不過 bar;**best-of-N + cost-select 推論縮放陡**(s4 8.86→s32 7.33,未平緩)→ 0709 判定應以縮放後可實現分評估;tail 獎池定價:n≥100 佔 77%(0.1358/0.1754)。

### 負面清單(不可投資;新增項用文獻/讀碼標記)

- **RL 座標 placement 取代 SA backbone**(F8,3-0)。
- **CBL/O-tree「構造合法性」假設**(0-3 否決:拓樸表示解碼不保證合法,legalizer 兜底不可省)。
- **「HPWL 強 surrogate」為 V10 proxy 背書**(0-3 否決)。
- **FloorSet 自產 generator**(讀碼:不存在,新專案級成本)。
- 既有量測判死(勿重議):GT-seed 三重判死、D1/A/B/M2/M3/M1c/N1/N2、N3 收案、Q2 規則 INVALID、Q4 已內建、Q5 無標的、pair-head-v2 幾何標籤公式。

---

## 4. 對「模型輸出利用方式」問題的直接回答

使用者假設:「graph encoder + coarse 生成 + guidance + exact repair,而非端到端;group/MIB/boundary 不全丟給 denoiser,guidance 與 repair 負責最後一公里」——**文獻證據(F5,3-0×4)支持,且本 repo 證據要求一個更強的特化**:

1. **分工正確**:ChipDiffusion 把 quality/legality 全放 denoiser 外(training-free guidance + post-hoc 合法化);FlowPlace 更進一步把 projection 放進 sampler 每步(構造性零重疊)。兩者都不讓 denoiser 學 constraints。
2. **特化修正——介面是 ORDER 不是座標**:本 repo 三重判死座標 seed 通道(GT 座標都推不動 SA backbone <0.002);唯一能兌現的模型輸出是 **golden-like 的 pairwise order**(golden order → 1.0535,勝 GT 直評 1.1079,因 boundary snap 連 golden 自帶的牆違規都修掉)。所以「coarse 生成」的正確產物是 order/pairwise 結構——座標只是 order 的載體(經 `build_order_dags` 抽取)。
3. **最後一公里的歸屬(D1 屍檢的教訓)**:decoder 自帶的 constraint pass(boundary snap + cluster 剛性)**弱於** production refine 疊層——soft-violation 稅(v_rel 翻倍)正是 D1 的死因。所以「repair 負責最後一公里」在本 repo 的正確實作是:**order-faithful realizer(longest-path compaction + exact-area)產合法幾何,production refine 疊層(slack/aspect/vsnap+guards)做 constraint 最後一公里**(排序表 #4)。
4. **訓練方法的對應結論**:(a) 監督換成 evaluator-蒸餾 ranking(F2;繞開 noisy fp_sol 與標籤循環);(b) head 訓練全程凍結 encoder/anchor(0706 鐵律);(c) 資料側近期只剩 scale/curriculum/n 覆蓋與 clean-sample 過濾(F6 斷崖 + 讀碼:generator 不存在);(d) offline-RL/hindsight BC(F7)只作 order 標籤基建,不是獨立路線。

---

## 5. Open questions(下一步的量測對象)

1. **Gate 0 兩變體的實測結果**——整條 ML-order 路線的生死判據(零件全在,一次 eval 級成本)。
2. clean-sample strict 子集的實際 clean ratio(若 >~10% → clean-only pair 訓練值得一測,仍受 Gate 0 封頂)。
3. E2 的 tail-5 paired delta;E1 pre-check 的 realizable 生成器選擇。
4. scale probe ep5-6(0709)——決定 #5 停車場是否開門。

## 6. 引用與工件

- 驗證後報告 JSON:session scratchpad `deep_research_result.json`(22 來源、8 findings、2 refuted、stats)。
- 兩份審查全文:session tasks `aa63547358a9d28ff`(deep-reasoner)、`a2b088c174bad355e`(Codex)。
- 文獻:見 §1 表(全部一手來源,3 票對抗驗證)。
- 本地證據:`docs/experiments/2026-07-05-column-backbone-pivot.md`、`2026-07-07-{d1-decoder-polish-probe,b-sp-window-packer,n1-area-scale-n2-tail-violations,n3-sa-throughput-scale-pt2}.md`、`.claude/plans/2026-07-07-order-decoder-roadmap.md`。
