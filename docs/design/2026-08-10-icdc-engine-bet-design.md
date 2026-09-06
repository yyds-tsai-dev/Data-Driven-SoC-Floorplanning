# IC/DC data-free cost 引擎大注 — Phase-0 完整設計

日期 2026-08-10|分支 `5.6-sol-reduce-time` @ 59686c4|作者:deep-reasoner
性質:設計文件 + 一次輕量離線量測(未跑 full-100 eval、未改 `partner/` 交付檔)
量測腳本與資產:`/nashome/NVL4/vdalab/yyds-dev/.claude/jobs/8b13fef4/tmp/icdc_assets/`

---

## 0. 決策摘要(先講結論)

1. **本注的問題結構被本次量測大幅簡化了。** 消費端(通道)不再是未知數:把「修復後 golden」(原生 noRT **1.0113**)注入通道 B + `PARTNER_REFINE_GUARD`,full-100 = **1.0695**,而 1.0695 恰好等於「覆蓋帶(權重 0.778)拿 golden 品質 + 未覆蓋帶(n≤102,權重 0.222)維持 control 品質 1.28」的加權和(反推 q_unc = **1.274**,由 §1 分帶表獨立算出 **1.284**,兩者吻合)。⇒ **覆蓋帶的通道磨損在 GUARD 下確實 ≈0,天花板 1.0695 完全由「22% 權重沒被注入」解釋。**
2. **因此 KPI 1.01 是良定義且結構上可達的**:若引擎(a)品質 ≈ 修復後 golden、(b)覆蓋全部 100 案、(c)GUARD 開啟,則 `full-100 noRT ≈ 引擎自身加權成本`。整注的風險**全部**集中到單一純量:**引擎自身輸出的加權成本 q_engine**。這是本設計最重要的結論——它把一個模糊的「換引擎」賭注化約成一條可量測的規格線。
3. **公式化決定:DiOpt 兩階段(direct_v2 warm start → data-free energy fine-tune)**,少步可微採樣 + **拓撲凍結可微合法化層(TFDL)** + group soft-min 目標。理由與替代案見 §3。
4. **本次量測翻掉了一個關鍵前提:α 內插谷不是「引擎品質→分數」的證據。** 實測 α 階梯每一階(含號稱 overlap-free 的 L 階梯)**全部 infeasible、V_rel 0.73–0.93**(golden 0.005、我方 production 修復後 0.015–0.035)。谷底不是「模型變好反而變差」,是「內插物是非法垃圾」。**α 谷不推翻 imitation 判死(那有 realize 2.35 等獨立證據),但它對「自洽引擎」零證據力 ⇒ Gate-0 必須用真正自洽的合法版圖重跑。**
5. **另一個翻案級發現:raw direct_v2 預測的 hpwl_gap ≈ 0.0009、area_gap ≈ 0.014(比 golden 還好),壞的只有 V_rel 0.81 與 overlap。** ⇒ 天真的「energy = hpwl + area」在退化重疊解上**已經是最小值**,模型今天就住在那個退化解裡。**能量必須在合法化投影之後求值**,否則訓練只會強化現有的重疊糊團。這直接決定了 §4 的架構。
6. **Gate-0 幾乎是免費的且已備妥九成**:前 session 建好但**從未評測**的 `alpha_L*.json` 階梯、修復後 golden `gr_layouts3.json`(1.0113)、我方 production 修復版 `gr_prod_layouts.json`(**1.1435**,100/100 feasible)都在手上;缺的只有「多樣性保全注入」的 15 行 harness 補丁。
7. **誠實的總成功率:達成 full-100 ≤1.14(可促轉)≈ 20%;達成 1.01 ≈ 3%。** 主風險不在通道、不在訓練基建、不在 GPU,而在「一個前饋模型能否在 ~50ms 內產出成本 <1.16 的版圖」——我方同族模型今天的合法化後品質是 2.3 級,差距是量級而非百分比。
8. **兩則前提更正(必須先修正對上層的認知)**:GPU 是**共用 NVIDIA L4 23GB**(非 A100 80GB),1M 訓練集**在本機**(`FloorSet/floorset_lite/`,24GB,100 workers × 90 檔,不需下載)。**final submission 日期在 repo 內任何官方文件中都不存在**,必須向官方/使用者確認(§7 的時程以 T+ 天表示)。

---

## 1. 前提盤點(全部本次一手核實)

| 項目 | 文件/簡報前提 | 實測 | 影響 |
|---|---|---|---|
| GPU | A100 80GB | **NVIDIA L4, 23034 MiB, 共用**(另有 DREAMPlace 佔 2.3GB);`direct_diffusion_train.py:21-26` 明文「the L4 is shared」,預設 `--vram-fraction 0.45`、`--gpu-util-cap 0.75` | 訓練 batch/K 要縮;但 §7 估算顯示 GPU **不是**瓶頸 |
| 1M 訓練集 | 「LiteTensorData 1M 實例」 | **在本機**:`FloorSet/floorset_lite/`,24 GB,100×90 檔(目錄名不叫 LiteTensorData,易漏) | 無下載成本;data-free 只需 `input_data`,與 `label_data(tree_sol, fp_sol, metrics_sol)` 完全分離(`FloorSet/lite_dataset.py:76-88`) |
| CPU | 48 核 | **64 核**,load avg 9.0,無他人重載 | eval 鏈與訓練 dataloader 需分時,見 §7 |
| final deadline | 未知 | **repo 內完全沒有**;只有 beta 7/31 17:00 與 beta 重傳 8/12 23:59(GMT+8) | **阻塞項:必須確認**;本文用 T+ 天 |
| 注入 harness | `PARTNER_ORACLE_PRED_FILE` 現成 | 現成但**只支援每案一張版圖並複製 K 份**(`column_sa_legalizer.py:3660-3664`) | **多樣性被摧毀 ⇒ 舊 α 量測全帶 +0.035 同質化稅**;Gate-1 前必須補多版圖格式 |

**官方成本函數(逐行核對 `FloorSet/iccad2026contest/iccad2026_evaluate.py:306-342`)**

```
Cost = (1 + 0.5·(max(0,hpwl_gap) + max(0,area_gap))) · exp(2.0·V_rel) · max(0.7, R^0.3)
V_rel = (V_boundary + V_grouping + V_mib) / N_soft
```
`ALPHA=0.5, BETA=2.0, GAMMA=0.3`(:72-74)。邊際比 `dC/dV_rel : dC/dgap = 2·C : 0.5·exp(...)` ≈ **4×**,與 memory 的 R0.7 結論一致。**取對數後能量天然可加:`log C = log(1+0.5(g_h+g_a)) + 2·V_rel`,4× 權重與乘性交互都自動正確。**

**0.3s 檔赤字分帶(引自 0807 備忘錄,本文據以做覆蓋率反推)**

| 帶 | 權重 | 赤字 | 反推該帶 per-case 成本 |
|---|---|---|---|
| 21-59 | 0.006 | 0.0019 | 1.32 |
| 60-89 | 0.069 | 0.0250 | 1.36 |
| 90-99 | 0.098 | 0.0297 | 1.30 |
| 100-109 | 0.226 | 0.0324 | 1.14 |
| 110-120 | 0.600 | 0.0801 | 1.13 |

---

## 2. 通道天花板的封閉式解(本文核心新結果)

設引擎覆蓋權重 `w_cov`、覆蓋帶引擎品質 `q_eng`、未覆蓋帶維持 control 品質 `q_ctl`:

```
full-100 noRT ≈ w_cov · q_eng + (1 − w_cov) · q_ctl        (GUARD 開啟,磨損≈0)
```

代入 G-T3-1 + GUARD 實測 1.0695、`q_eng = 1.0113`、`w_cov = 0.778`(n≥103):

```
q_ctl = (1.0695 − 0.778×1.0113) / 0.222 = 1.274
```

由上表獨立加權 n≤102 各帶(0.006×1.32 + 0.069×1.36 + 0.098×1.30 + 0.049×1.14)/0.222 = **1.284**。**兩條互相獨立的路徑吻合到 0.8%** ⇒ 模型成立(A 級)。

**三個立即可用的推論:**

- **P1(覆蓋率是與品質同量級的槓桿)**:`d(noRT)/d(w_cov) = q_ctl − q_eng ≈ 0.26`。把覆蓋率從 0.778 拉到 1.00(即讓 n≤102 也吃引擎)在完美品質下值 **−0.058**。這比整個 0806 促轉包(−0.0095)大六倍。現行阻擋點是 `PARTNER_DIRECT_MIN`(`contest_optimizer.py:160`,production 覆寫為 0.3)與 rung-0 預算投影 `_direct_seat_ts`。
- **P2(GUARD 從「判死」升格為「前置基建」)**:0810 裁決 GUARD「一般輸入 0/115 次取代=空操作,不促轉為分數手段」——正確,但它在**高品質輸入**下把磨損從 +0.0211 打到 0.0000。引擎線一旦成立,GUARD 是**必要條件**而非可選項。目前 default off、15 tests、位元級 off-path 安全,狀態剛好。
- **P3(選擇器不是瓶頸)**:T1 實測 G1 = 0.0005、承重帶 21/21 無誤差 ⇒ 「引擎候選 vs column 冠軍」的 per-case min-selection **實質等於 oracle**。⇒ **Mode B 整合在期望值上是下檔封閉的**(引擎輸的案子會被選擇器丟掉),唯一代價是它吃掉的 pool 席次與時間。

**由此得到引擎的規格線(Gate-1/2 的判準來源):**

| 引擎自身加權成本 q_eng | 覆蓋 0.778 的投影 | 覆蓋 1.000 的投影 | 判讀 |
|---|---|---|---|
| 1.011(=修復後 golden) | 1.0695 | **1.011** | KPI 達成 |
| 1.05 | 1.100 | 1.050 | 大勝 |
| 1.10 | 1.139 | 1.100 | 可促轉 |
| **1.16(=現役 1.1647)** | **1.185** | **1.160** | 損益平衡點(0.778 覆蓋下反而虧) |
| 1.30 | 1.294 | 1.300 | 明確判死 |

**⚠ 損益平衡的門檻極高:0.778 覆蓋下引擎必須 q_eng ≤ 1.086 才打平現役 1.1647。** 這是「先把覆蓋率修到 1.0 再談引擎」的量化理由——覆蓋率修好後門檻鬆到 q_eng ≤ 1.16,而 1.16 剛好落在我方 3.5s 檔求解器(1.118)與 0.3s 檔(1.165)之間,是一個「難但不荒謬」的目標。**⇒ 覆蓋率修復(P1)是 Gate-1 的前置工作,不是 Gate-3 的整合細節。**

---

## 3. Q1 公式化決策

### 決定:**F3 — DiOpt 兩階段(supervised warm start → data-free energy bootstrap)**

具體配方:以 `direct_v2` EMA 權重(`partner/checkpoints/direct_v2_cont/eval_step1p2M.pt`)為起點,用 **S=2–4 步可微採樣**(對齊生產 `PARTNER_DDIM_STEPS=2`)直接對能量做反傳(DRaFT 式直接梯度,**不走 RL**),目標為 **group soft-min**,能量在 **拓撲凍結可微合法化層之後**求值。

### 理由

| 判準 | F1 攤提回歸 | F2 從零訓 exp(−βE) sampler | **F3 DiOpt(採用)** |
|---|---|---|---|
| 時程(T+7~21 天) | 最短(3-5 天) | **不可行**(rKL/SN-NIS 從零,L4 上 3-6 週) | 中(warm start 省掉全部表徵學習) |
| 我方基建 | 需新模型 | 需新模型 + 新 sampler | **`fast_condition`(26+9 維特徵)、`DirectDenoiser`、`sample_direct_dpmpp`、`z_to_rectangles` 全部現成且與生產推理共用** |
| α-谷/自洽性要求 | **致命**:確定性回歸對多模態解做 mode averaging = 產出正是內插式糊團 | 佳(採樣自單一模態) | 佳(擴散採樣本質上抽單一模態,不做模態平均) |
| 多樣性保全 | **零**(每案一解) | 佳(溫度可控) | 佳(不動雜訊注入,只改去噪器;K 樣本天然多樣) |
| 與 2 步推理的匹配 | n/a | 差(1000 步訓 → 2 步採樣失配) | **佳**(訓練即用 2-4 步,訓練/推理同分佈) |
| reward 可微性需求 | 需可微 | 不需 | 需可微 → **§4 的能量就是為此設計** |

**為何不走 Flow-GRPO / policy gradient**:GRPO 只在 reward 不可微時才必要。我們把能量設計成版圖的可微函數後,直接梯度的樣本效率高 1-2 個數量級,且無 KL 係數的「大則無增益、小則塌縮」兩難。**保留為 Gate-2 stretch**:若 §4 的可微能量與「注入後真實分數」相關性不足(Gate-0.5 會量),再用 GRPO 以真實注入分數當 reward 補一小段——但那時 reward query = 一次完整求解,成本高一個量級。

**替代案(若 F3 在 Gate-1 訓練不穩)**:降級為 **F1 + 拓撲頭**——不回歸座標,而是回歸「欄位指派 + 欄內順序」這種離散結構,再由現成 column legalizer 具現化。它繞開 mode averaging(離散結構空間裡的平均仍是合法結構),但 0707「order-only 通道 KILL」(golden order × 非 golden shapes = 9.2-9.4)已對純 order 通道判死,故僅在 (order, shape) 聯合輸出下才成立。列為 B 案,不預設投入。

---

## 4. Q2 Energy 定義

### 4.1 架構:**能量必須在投影之後求值**(本文第二個核心結論)

實測依據(§0 第 5 點):`alpha_pred0.json`(raw direct_v2 預測)的 `hpwl_gap = 0.0009`、`area_gap = 0.014`——**比 golden 還好**,因為所有塊擠成一個重疊糊團時線長與外框都最小。若能量寫成 `w_h·hpwl + w_a·area + w_ov·overlap`,模型只要調高一點 overlap 就能無限降低前兩項;梯度會把它推回它今天已經在的退化解。

**解法:拓撲凍結可微合法化層(TFDL)**

```
x̂0(座標+log aspect)
  → [離散、detach] 由 x̂0 抽出每對塊的分離軸與方向 → DAG 邊集 E_x, E_y
  → [可微] 沿 E 做最長路徑壓實(longest-path compaction),前向推 + 後向拉
  → 合法版圖(overlap-free by construction, 面積精確)
  → Energy(合法版圖)
```

- 「固定拓撲後壓實是線性規劃」是經典結果;梯度對座標與尺寸幾乎處處存在(argmax 分支 detach)。
- **現成參考實作已在手上**:`icdc_assets/alpha_legal.py:project()`(前向/後向兩趟拓撲掃描)與 `gr_lib.py:build_topology/longest_path_bounds/solve_axis`。移植成 torch 版即可。
- 這同時解釋並修正了 noise-opt 的敗因:noise-opt 最小化的是 **raw overlap**(refine 免費提供的東西);TFDL 之後 overlap 恆為 0,能量剩下的全是 **refine 修不好的東西**(hpwl 拓撲最優性 + V)。**這是機制區隔的技術核心。**

**成本**:n≤120,每軸 O(n²) 建邊 + O(n) 拓撲掃描;GPU 上是 ~2n 次小 kernel,估 5-15 ms/batch。可接受。
**退路**:若 TFDL 實作超時,用「非可微生產 legalizer + straight-through(恆等梯度)」;梯度品質差,但一天可落地。

### 4.2 能量公式(evaluator-faithful)

```
E(layout) = log(1 + 0.5·(ĝ_h + ĝ_a)) + 2.0·Ṽ_rel + λ_ov·Ω
```

| 項 | 定義 | 備註 |
|---|---|---|
| `ĝ_h` | `leaky(hpwl/hpwl_ref − 1, slope=0.05)`,hpwl = **質心 Manhattan**,逐行對齊 `calculate_hpwl_b2b/p2b`(:180-195) | evaluator 用 `max(0,·)`;改用小斜率 leaky 是刻意取捨——保住已優於 golden 的樣本的梯度訊號,代價是輕微不忠實 |
| `ĝ_a` | `leaky(bbox_area/area_ref − 1)`,bbox 用 hard max/min(次梯度) | 同上 |
| `Ṽ_rel` | `(Ṽ_bnd + Ṽ_grp + Ṽ_mib) / N_soft`,**N_soft 逐行照抄 evaluator :459-471** | 見 4.3 |
| `Ω` | TFDL 之後恆為 0;僅在 straight-through 退路下才需要 | λ_ov 只在退路生效 |

**normalization 不依賴 golden 的處理**:`hpwl_ref / area_ref` 只出現在 **loss**,永不進入模型輸入 ⇒ 推理時不需要,不構成 imitation(它是純量正規化,不是座標目標)。訓練時可直接取 `label_data` 的 `metrics_sol`(1M 樣本全有)。
**替代案(純粹主義)**:改用實例內生尺度 `hpwl_ref := Σ_e w_e · √(Σa)`、`area_ref := Σ a_i / 0.97`,不碰 label 任何欄位。代價:每實例權重與官方定標偏離,clamp 位置錯位。**建議主用 golden 純量、同時在 Gate-2 跑一組內生尺度對照**,因為 hidden 泛化紀律偏好不依賴任何 golden 訊息的配方。

### 4.3 硬約束:能結構化的一律結構化(不進 loss)

| 約束 | 處理 | 出處 |
|---|---|---|
| **面積 1% 雙側硬約束** | **by construction**:z 表示為 `[x/S, y/S, log(w/h), ·]`,`w=√(A·e^a), h=√(A/e^a)` ⇒ `w·h ≡ A` 恆等 | 現成:`direct_diffusion_model.py:21-23`、`layout_losses.py:36-50` |
| **fixed 形狀 / preplaced 位置** | **by construction**:`known_z_channels()` 在每個採樣步把已知通道覆寫回去 | 現成:`direct_diffusion_model.py:185-210, 238-247` |
| **MIB 同形** | **by construction(新增)**:對每個 MIB group 把 `z[...,2]` 取組內平均後再解碼 ⇒ `V_mib ≡ 0` | 一行 scatter-mean;把整個 mib 桶從能量裡消掉 |
| **overlap** | **by construction**:TFDL 投影 | §4.1 |
| **boundary** | 軟項 `Ṽ_bnd` | 下 |
| **grouping** | 軟項 `Ṽ_grp` | 下 |

**軟項必須是「計數型」而非「距離型」**(對現有 `layout_losses.py` 的實質升級):evaluator 數的是**離散 bit**,距離型 loss 會把梯度浪費在把一個已經違規 10% 的塊拉到違規 5%(分數上零收益)。改用飽和替代:

```
Ṽ_bnd = Σ_i [1 − exp(−d_i/τ)] ,  d_i = Σ_{bit∈code_i} |edge_i − bbox_edge|   (τ ≈ 1e-3·√Σa)
Ṽ_grp = Σ_g Σ_{i∈g} [1 − max_{j∈g, j≠i} c_ij] + μ·Σ_g (bbox_g 面積 − Σ_{i∈g} a_i)/Σ_{i∈g} a_i
          c_ij = exp(−relu(sep_x)/τ)·σ(ovl_y/τ) + exp(−relu(sep_y)/τ)·σ(ovl_x/τ)
```
- `Ṽ_bnd` 的底層距離即現成 `layout_losses.boundary_loss`(:96-118),只需外掛飽和。
- `Ṽ_grp` 兩項分工:compactness 項提供長程梯度(把散開的群拉近),max-contact 計數項提供短程、計數形狀的梯度。**誠實標注:這是全套裡最不忠實的鬆弛**(真值是連通分量數);grouping 是我方唯一顯著劣於 golden 的軸(47 vs 10),但在計分帶只佔 11 條,故不擋整體。

### 4.4 gr_lib 修復器的定位(明確裁決)

- **不放進訓練迴圈**(每步一次 CPU LP = 吞吐殺手,且會讓 GPU 空轉)。TFDL 已在 GPU 上提供「投影」語義。
- **放在推理後處理**:引擎輸出 → (可選)`gr_lib` 風格保序聯立 LP 精修。但 0806 已裁決完整 A+B+C 為 0.68s/案 = 2.4× 全預算,**不促轉**;存活切片 `PARTNER_COORD_POLISH`(scipy-backed,+0.0565s)已產品化,**引擎輸出直接沿用它即可**,零新成本。
- **Gate-2 stretch**:把 `gr_lib` 的 boundary 反轉 + grouping 橋接當作「訓練後一次性資料重標」——不是把它放進 loss,而是拿它把 1M 的 golden 修乾淨後,**當作 Gate-2 的 hold-out 品質參考線**(不是 imitation 目標)。

---

## 5. Q3 多樣性 / 塌縮防禦

**問題規模已量化**:`α=0 注入(K 份同一張預測)比 control 差 +0.035`。這 0.035 **不是模型品質訊號,是同質化稅**。舊 α 曲線每一階都繳了這筆稅,故其形狀被系統性扭曲;扣掉後 α=0≈control、α=0.5≈+0.022、α=0.75≈+0.001、α=1≈−0.104。谷仍在,但淺得多,且交叉點在 α≈0.8 而非 0.9+。

**三層防禦:**

1. **目標函數層(主防禦)**:group soft-min。每實例抽 K=8(不同雜訊),
   `L = −(1/β_g)·logsumexp(−β_g·E_k)`,β_g 調到有效權重集中在前 1–2 名。
   直接 `min` 只給 argmin 梯度、其餘樣本失訓;soft-min 讓劣樣本拿到小權重梯度,同時**不獎勵「每個樣本都一樣好」**——這正是 vBoN / inference-aware BoN 的性質,GRPO 的 group-relative 只是它的 RL 版本。**多樣性在這個目標下是資產而非要罰的副作用。**
2. **參數層**:對 base 權重的 L2 錨(或對同一雜訊下 base 模型 x0 的輸出空間 KL),係數以「group spread 不掉」為調參依據,不以 loss 為依據。
3. **監控層(死刑條款的觸發器)**:每 N 步記錄 `spread_K = mean_{k<l} ‖x0_k − x0_l‖ / √Σa`。
   **停訓條款:`spread_K < 0.7 × spread_K(base direct_v2)` 連續 2000 步 ⇒ 立即停訓,判為塌縮,回退上一個 snapshot 並降 β_g / 升錨係數。**
4. **消費層**:注入必須是 **K 張不同版圖**(見 §9 的 harness 補丁),否則量測會重蹈 0.035 稅。溫度旋鈕在推理側保留(採樣雜訊尺度),但 **Gate-1 一律用 base 溫度**,避免多一個混淆因子。

---

## 6. Q6 與 noise-opt / reward-FT 的機制區隔(書面化)

0805 survey §4 的立場是:reward-FT ≡ noise-opt 的離線攤提版,同通道、同死因,**預設不投資**。本設計必須逐條回應,而不是宣稱免疫。

| 軸 | noise-opt(0723 判死,+0.1175) | 單樣本 reward-FT(survey 預設不投) | **本設計** |
|---|---|---|---|
| 優化對象 | per-case 推理期對 input noise 做梯度下降 | 攤提進權重,單樣本 reward | **攤提進權重,group soft-min reward** |
| 推理期成本 | **付費**(且從 SA 手上偷時間 ⇒ 雙重虧) | 零 | **零**(權重內,推理不變) |
| 目標量 | **raw overlap 能量** = refine 本能免費提供的東西 | raw 品質 | **TFDL 投影後的官方成本**(overlap 恆 0,目標只剩 refine 修不好的 hpwl 拓撲最優性與 V) |
| 多樣性 | 同質化(明確敗因) | 塌縮整個先驗分佈(更糟) | soft-min 獎勵多樣性 + spread 停訓條款(§5) |
| 消費端證據 | 當時未知 | 未知 | **G-T3-1 已證通道會消費好 prior 且更快(0.268 vs 0.293s);§2 封閉式解證明覆蓋帶磨損 ≈0** |
| 下檔 | 開放(偷時間) | 開放 | **封閉**(P3:選擇器實質 oracle,引擎輸的案被丟掉;唯一代價是席次) |

**誠實聲明**:上表第五列(消費端證據)是唯一真正承重的區隔,而它目前只在 **q_eng ≈ 1.011 的極端點**被證實。中間區段(q_eng ∈ [1.05, 1.20])**完全沒有證據**——Gate-0 存在的唯一理由就是把這段曲線量出來。若曲線在該區段非單調或交叉點落在 q_eng < 1.05,本設計與 noise-opt 的區隔就**不成立**,整注當場結案。

---

## 7. Gate 鏈(fail-fast,含死刑條款)

**⚠ 全鏈量測紀律(0807/0810 成文,不得例外)**:①判讀一律用加權 `total_score_no_runtime`;②促轉級判準 3-rep ≥0.008 或 ≥8 reps,每鏈強制 inert null 臂(單跑 sd ≈0.0047);③跨 session 絕對值不可比,只信同 session 成對;④eval 期間禁止任何代理編輯 solver 檔(用 worktree 隔離);⑤量測前查 `ps`/`who`,標注污染窗。

| Gate | 內容 | 工時 | 通過判準 | **死刑條款** |
|---|---|---|---|---|
| **G0-a** 曲線(先) | 用**已存在**的自洽合法版圖建品質階梯並注入:`gr_layouts3`(1.0113)/ 3.5s 檔 winner dump(~1.118)/ `gr_prod_layouts`(**1.1435**,100/100 feasible)/ 0.3s 自我注入(1.1647,恆等控制)。每階 ×3 rep 成對,GUARD on/off 雙臂,含 null 臂 | **1 天** | 注入分數沿自洽流形**單調遞減**,且交叉點(注入分 = control)對應 `q_eng ≥ 1.10` | **交叉點 q_eng < 1.05 ⇒ 整注結案**(引擎需比修復後 golden 還好才有用,不可能);**非單調(如 1.1435 階注入比 1.1647 階差)⇒ 整注結案**,並把「通道只吃近完美輸入」寫入判死清單 |
| **G0-b** 覆蓋率 | P1:量 `PARTNER_DIRECT_MIN` / `_direct_seat_ts` 放寬到 n≤102 的**時間**代價(不需引擎,用現役預測即可) | 0.5 天 | 覆蓋率 0.778→≥0.95 時 avg runtime 增幅 ≤0.03s 且 noRT 不劣化 >0.005 | 若放寬覆蓋率必然使 avg runtime 破 0.35s ⇒ **引擎只能吃 0.778 覆蓋 ⇒ G0-a 的交叉點門檻收緊到 q_eng ≥ 1.086,幾乎必然觸發 G0-a 死刑** |
| **G0.5** 能量保真 | 實作 `energy.py` + 逐行對拍 evaluator;在 100 案 × 多個現成版圖檔上比對 | 1 天 | 對官方 per-case cost 的 Spearman ≥0.95;`Ṽ_bnd/Ṽ_grp` 對官方計數 rank corr ≥0.8;TFDL 輸出 100/100 feasible | Spearman <0.9 ⇒ 能量重設計(一次);二次失敗 ⇒ 降級為 GRPO(reward=真實注入分),成本 +1 週,需重新上呈 |
| **G1-a** 訓練健檢 | 從 direct_v2 熱啟,在 **1k 訓練實例**(n∈[95,120],非驗證 100 案)上做 energy fine-tune,S=2,K=8 | 2 天 | 訓練集上 `E` 單調下降;TFDL 後 V_rel 從 ~0.8 降到 <0.10;`spread_K` 未觸發停訓條款 | 5000 步內 V_rel 未降到 <0.3,或 spread 觸發停訓兩次 ⇒ **判定 F3 不穩,轉 §3 的 B 案(離散結構頭)或結案** |
| **G1-b** 注入實分 | 用 G1-a 的 checkpoint 對 100 驗證案各出 **K=8 張不同版圖**,經**多版圖 harness** 注入,3-rep 成對 + null 臂,GUARD on | 1 天 | ① 原生 `q_eng` 落在 G0-a 交叉點以下;② full-100 注入分 **≤ control − 0.015** | 注入分 > control − 0.008(即落在噪音內)⇒ **本注結案**,預算歸還 R6/R7 線 |
| **G2** 規模化 | 1M 全量 data-free 訓練 + hold-out 泛化(訓練集 instance 的 hold-out split,**不是**驗證 100 案) | 5-7 天 | hold-out `E` 與 train `E` 差距 <10%;full-100 注入分 ≤ control − 0.030;runtime 不增 | hold-out 明顯劣於 train(過擬合 in-distribution)⇒ 停,只留 G1 規模模型做小幅促轉評估;full-100 未達 −0.030 ⇒ 不進 G3 |
| **G3** 整合/交付 | Mode B 產品化(引擎當額外候選、min-selection、GUARD 常開、覆蓋率修正)、frontier-括號成對 gate、final repack | 2-3 天 | frontier 校正淨值 ≤−0.015(3-rep)或 8-rep ≥3σ;100/100 feasible ×9;max_rt 在官方免費上限內;解壓後 tar 跑 full-100 復現 | 任一不過 ⇒ 旗標 default off 保留為研究儀器,**不進提交包** |

**排程(T = 開案日;final deadline 待確認,以下假設 ≥T+21)**

```
T+0   G0-a 資產搬遷 + harness 多版圖補丁 + G0-b 覆蓋率量測      ← 阻塞全鏈
T+1   G0-a 執行(~15-20 runs @ ~40s,含 3.5s 檔 dump)          ← 第一個死刑點
T+2   G0.5 energy.py + 對拍                                     ← 第二個死刑點
T+3   TFDL torch 實作 + 單測
T+4~5 G1-a 小規模訓練
T+6   G1-b 注入實分                                             ← 第三個死刑點(最重要)
T+7~13 G2 1M 訓練(GPU 背景跑,期間 CPU 可並行做 R6/R7)
T+14~16 G3 整合 + frontier gate
T+17~19 final repack + 解壓驗證(0806/0810 兩次都在這裡抓到出貨 bug,不可壓縮)
T+20~21 緩衝
```

**資源**
- **GPU(L4 23GB 共用)**:直接梯度、S=2、batch 4 實例 × K=8 = 32 版圖 × N≤120 token,d512/l12 ⇒ 前向 ~50 GFLOP,含反傳與 2 步 ~0.4 TFLOP/step。L4 bf16 實效 ~25 TFLOPS、util cap 0.75 ⇒ **~50-150 ms/step**;G1-a 5k 步 ≈ 10-20 分鐘,G2 200k 步 ≈ **3-8 天**(與 direct_v2 的 1.2M 步同量級,故 G2 必須背景長跑並支援中斷續跑——沿用 `train_forever_claude.sh` 模式)。**GPU 不是瓶頸,但 G2 的牆鐘是排程主導項。**
- **CPU(64 核)**:每次 full-100 @0.3s ≈ 30-40s + 載入;整個 gate 鏈的 CPU 成本微不足道。**真正的衝突是 dataloader worker 與 24-worker SA pool 搶核** ⇒ 紀律:**eval 窗內把訓練程序 `SIGSTOP`**(或 `--num-workers 2` + nice),並在每次 eval 前記錄 `uptime`。
- **磁碟**:`floorset_lite` 24GB 已在;checkpoint 每個 1.7GB × keep-recent 3 + snapshots ⇒ 預留 30GB(`/nashome` 尚餘 12TB,無虞)。

---

## 8. 誠實的成功率評估

| 環節 | 我的估計 | 依據 / 主要不確定性 |
|---|---|---|
| G0-a 通過(曲線單調且交叉點 ≥1.10) | **40%** | 正面:§2 封閉式解、G-T3-1 更快收斂。負面:交叉點取決於「席次被引擎吃掉」的機會成本,而現行 NREF=6/24;完全沒有中間點資料 |
| G0-b 覆蓋率可修 | 65% | `DIRECT_SEAT_FIX` 已證 rung-0 預算投影可算;成本主要是 sampler 冷啟(0810 已量 tid78 0.90→0.67s) |
| G0.5 能量保真 | 80% | 官方公式短且已逐行讀過;風險集中在 grouping 鬆弛 |
| G1-a 訓練穩定(V_rel <0.10) | 50% | TFDL 之後 V 是純幾何目標,梯度乾淨;但 boundary 需要「全域 bbox 邊」的協同,擴散模型在 2 步內能否協調 100+ 塊未知 |
| G1-b 達 −0.015 | **25%**(條件於前四關) | **最大不確定性**。要求 q_eng ≲ 1.10–1.16,而我方同族模型合法化後歷史品質是 **2.35**(scale probe realize)。差距是量級。反面樂觀證據:那是 imitation 訓練的模型,從未以「合法化後成本」為目標訓練過——這正是 data-free 的賣點,但屬未驗證假說 |
| G2 泛化 | 60% | 1M in-distribution,hidden 與 public 同分佈(官方說明) |
| G3 促轉 | 70% | 下檔封閉 + 選擇器 oracle;風險在 runtime |
| **綜合:full-100 ≤1.14 且促轉** | **≈20%** | 0.40×0.80×0.50×0.25 ≈ 0.04 為嚴格串聯下界;考慮到部分關卡可退而求其次(如只吃覆蓋帶、只促轉部分帶),上修到 20% |
| **綜合:達成 KPI 1.01** | **≈3%** | 需 q_eng ≈ 1.01 **且** 覆蓋率 1.0。q_eng ≈ 1.01 意味一個前饋模型在 50ms 內贏過我方 3.5s SA 達 0.11——沒有任何文獻或內部證據支持 |

**與其他路線的機會成本**:同樣 3 週投在 R6(packing 收緊,dead space 10.3% vs golden 2.9%)+ R7(帶約束塊區域正確放置)+ P1 覆蓋率修復,我估合計 EV ≈ −0.02~−0.04(到 1.13-1.15),成功率 60%+。**若使用者的效用是「盡可能高名次」而非「賭 1.01」,R6/R7/P1 是更好的配置;若效用是「非 1.01 不可」,則只有本注(或別組架構情報)有非零機率。** 這個取捨應由使用者裁決,本文不代決。
**建議的折衷**:**G0-a / G0-b 兩天無論如何要做**——它們不需要引擎、成本 1.5 天、產出(交叉點 + 覆蓋率修復)對 R6/R7 線同樣有用,且 G0-b 本身可能是一個 −0.01 級的獨立促轉品。**先做這兩關,拿到交叉點再決定要不要押後面 3 週。**

---

## 9. Gate-1 可執行實作清單(檔案級)

原則:**新程式碼一律落在新目錄,不動 `partner/` 既有交付檔**(另有代理在改 `partner/`);唯一的 `partner/` 改動是 15 行 probe-only 補丁,且 off-path 必須位元不變。

### 9.0 前置(T+0,必須先做,~1 小時)

- [ ] **資產已搬遷(本次已做)**:`/tmp/claude-1100/.../c70131c1-.../scratchpad` 的 `gr_*`/`alpha_*` 共 40 檔已複製到
      `/nashome/NVL4/vdalab/yyds-dev/.claude/jobs/8b13fef4/tmp/icdc_assets/`(3.5MB)。**`/tmp` 隨時可能被清,原位置不可再依賴。**
- [ ] 把 `icdc_assets/` 內 `gr_lib.py` + `gr_repair3.py` + `alpha_legal.py` 三個檔提升為 repo 內資產:
      `scratchpad/icdc/`(不進 `partner/`,不進提交包)。
- [ ] 確認 final submission 日期(repo 內沒有),回報使用者。

### 9.1 harness 補丁(唯一的 `partner/` 改動,~15 行)

**`src/solver/column_sa_legalizer.py`** — 擴充 `oracle_pred_override`(:3636-3664):
- 新增 `PARTNER_ORACLE_PRED_MULTI=1`:JSON 值允許是 `[[ [x,y,w,h], ... ], ...]`(每案 **K 張不同版圖**)。
- 語義:`preds` 的前 `min(K, len(preds))` 項逐一替換為不同版圖(不再是 `[P.copy() for _ in preds]`)。
- **硬要求**:`_ORACLE_PRED_FILE` 為空時整段仍是 falsy 常數測試,production 路徑位元不變;新增 3-5 個單測(空檔、K<len、K>len、blockcount 不符 degrade 回 control)。
- 檔頭註解沿用現有的 `*** PROBE ONLY, NEVER PROMOTABLE ***` 標記。

### 9.2 新套件 `src/floorset_arch/icdc/`

| 檔案 | 內容 | 相依 |
|---|---|---|
| `energy.py` | `decode_rects(z, area, cons, tp)`(exact-area 參數化 + MIB aspect 綁定 + known-channel 覆寫)、`hpwl_centroid_manhattan()`、`bbox_area()`、`v_boundary_soft()`、`v_group_soft()`、`energy()`(§4.2 log-cost 形式)、`official_cost_reference()`(numpy,對拍用) | 逐行對齊 `iccad2026_evaluate.py:180-342, 427-540` |
| `legalize.py` | **TFDL**:`extract_topology(rects) -> (E_x, E_y)`(detach)、`compact(rects, E, sizes)`(前向推 + 後向拉,torch,可微)、`tfdl(x0) -> legal rects` | 參考 `icdc_assets/alpha_legal.py:project`、`gr_lib.py:build_topology/longest_path_bounds` |
| `sampler.py` | `sample_differentiable(model, cond, schedule, steps=2, K=8)` — 複製 `direct_diffusion_model.sample_direct_dpmpp` 的更新式但**移除 `@torch.no_grad`**、支援 K 樣本 batch、保留 anchor clamp | import `src/solver/direct_diffusion_model.py` |
| `train_energy.py` | fine-tune 迴圈:floorset_lite dataloader(**只取 `input_data`**)→ `fast_condition` → `sample_differentiable` → `tfdl` → `energy` → group soft-min + L2 錨 → AdamW + EMA;`spread_K` 監控與停訓條款;L4 etiquette 旗標照抄 `direct_diffusion_train.py:227-231`;SIGTERM 續跑 | import `partner/direct_diffusion_train.fast_condition` |
| `dump_preds.py` | 對 `LiteTensorDataTest` 100 案各抽 K 張,寫成 MULTI 格式 JSON | — |
| `__init__.py` | — | — |

### 9.3 測試 `tests/`

| 檔案 | 內容 |
|---|---|
| `test_icdc_energy.py` | ① `official_cost_reference` 對 `iccad2026_evaluate.evaluate_solution` 在 `gr_layouts3.json` 100 案上逐案吻合 <1e-9;② `energy()` 對官方 `cost_no_runtime` 的 Spearman ≥0.95(用 `gr_layouts3` / `gr_prod_layouts` / 3.5s dump 三組);③ `Ṽ_bnd` 對官方 `boundary_violations` rank corr ≥0.8 |
| `test_icdc_legalize.py` | TFDL 輸出在 100 案上 `check_overlap()==0`、`check_area_tolerance()==0`、`check_dimension_hard_constraints()==0`;梯度有限且非零 |
| `test_partner_oracle_multi.py` | 9.1 的 off-path 位元不變 + 四個 degrade 情境 |

### 9.4 量測腳本 `scratchpad/`

| 檔案 | 內容 |
|---|---|
| `icdc_native.py`(**已寫,已跑**) | 任意版圖 JSON 的原生加權 noRT + 分帶 V_rel/hg/ag。位置:`icdc_assets/icdc_native.py` |
| `icdc_gate0_chain.sh` | G0-a:5 臂(golden / 3.5s / prod-repaired / self / null)× GUARD on-off × 3 rep,Latin-square 交錯,沿用 `icdc_assets/alpha_arm.sh` 的 0.3s 操作點 env |
| `icdc_gate0_dump.sh` | 產生 3.5s 檔 winner 版圖 dump(G0-a 缺的那一階) |

### 9.5 Gate-1 之前必須先跑的三條指令(不需寫任何新程式)

```bash
# 1) 三個已存在版圖檔的原生分數(已完成,見 §0 與下表)
cd .../icdc_assets && uv run python icdc_native.py gr_layouts3.json gr_prod_layouts.json

# 2) 3.5s 檔 winner dump(補 1.118 那一階)—— 需要 icdc_gate0_dump.sh
# 3) G0-a 注入鏈 —— 需要 9.1 的 harness 補丁 + icdc_gate0_chain.sh
```

---

## 10. 本次一手量測結果(A 級,可用 `icdc_assets/icdc_native.py` 重算)

| 版圖檔 | 案數 | feasible | 原生 noRT | 說明 |
|---|---|---|---|---|
| `gr_layouts3.json` | 100 | **100** | **1.0113** | 修復後 golden;V_rel 0.004-0.006,hg/ag ≈0 |
| `gr_prod_layouts.json` | 100 | **100** | **1.1435** | 我方 0.3s 產出 + R1 完整修復;V_rel 0.015-0.035,110-120 帶 1.1076 |
| `alpha_pred0.json`(raw direct_v2) | 26 | **0** | 10.0 | **hg=0.0009, ag=0.014(優於 golden),V_rel=0.81** ⇒ 退化重疊解 |
| `alpha_L00/L25/L50/L75.json` | 26 | 1 | 9.63-9.76 | 號稱 overlap-free 的內插階梯**仍全數 infeasible**,V_rel 0.73-0.94 |
| `alpha_a00/a50.json` | 26 | 0 | 10.0 | 原始內插階梯 |

**判讀**:
1. 舊 α 曲線的每一階都是 **infeasible 且 V_rel ≈0.8 的物件**。它量到的是「非法垃圾裡摻多少 golden 座標」,不是「自洽引擎的品質」。**α 谷對本注零證據力**(但不推翻 imitation 判死本身,那有 realize 2.35 與 prior 負斜率兩條獨立證據)。
2. `L` 階梯 infeasible 的最可能原因是 preplaced 的 `(x,y,w,h)` 在 1e-4 容差下被投影破壞(`check_dimension_hard_constraints`);**任何引擎/投影層都必須把 preplaced 當硬凍結**,這是 §4.3 已納入的設計。
3. `gr_prod_layouts` = **1.1435 / 100% feasible** 是 G0-a 最理想的中間階(現成、零成本),它與 self-injection(1.1647)、3.5s dump(~1.118)、golden(1.0113)構成四點階梯。

---

## 11. 證據等級與未驗證項

**A 級(本次一手或逐行讀碼)**:§1 成本公式與常數;§10 全部原生分數;§2 的 `q_ctl` 雙路徑吻合;§4.3 的 by-construction 清單(逐行對到 `direct_diffusion_model.py` / `layout_losses.py`);GPU/資料/核心數盤點;harness 只複製單一版圖的事實(`column_sa_legalizer.py:3660-3664`)。

**B 級(引自既有實驗紀錄,未本次復現)**:G-T3-1 的 1.0932 / GUARD 1.0695;T1 的 G1=0.0005;0.3s 分帶赤字表;同質化稅 0.035;噪音地板 sd 0.0047。

**C 級(推估)**:§7 的 GPU 步時與 G2 牆鐘;§8 全部成功率;`Ṽ_grp` 鬆弛的忠實度;TFDL 在 GPU 上的實際延遲。

**明確未驗證 / 已知風險**:
1. **G0-a 的中間段完全沒有資料**——整份設計的承重假設。
2. `Ṽ_grp` 是連通分量數的鬆散上界,可能與官方計數脫鉤。
3. 少步(S=2)可微採樣的梯度是否足以訓練:DRaFT-K 文獻在影像域有效,**layout 域無先例**(0805 survey §2(d):EDA 域「reward 由下游 solver 決定」查無先例)。
4. 用 `metrics_sol` 當 normalizer 是否算「使用 golden 資訊」——法規上無疑(它不進模型輸入),但泛化紀律上建議在 G2 跑內生尺度對照。
5. **final deadline 未知**:§7 排程假設 ≥T+21;若實際 <T+14,**G2 不可能完成,整注應降級為「只做 G0-a/G0-b + R6/R7」**。
6. 覆蓋率修復(P1)與 `PARTNER_DIRECT_SEAT_FIX`(已量 −0.0031 官方分)方向相反(後者是「只關不開」),兩者需一併重新設計,不可各自促轉。

---

## 12. 上呈使用者的三個決策點

1. **要不要在拿到 G0-a 交叉點之前就承諾 3 週?** 建議 **不要**;先做 G0-a + G0-b(1.5 天,對 R6/R7 線同樣有價值)。
2. **效用函數是「盡可能高名次」還是「非 1.01 不可」?** 前者 ⇒ R6/R7/P1(EV −0.02~−0.04,成功率 60%+);後者 ⇒ 本注是唯一非零機率路徑(3%)。
3. **final submission 日期**——repo 內查無,阻塞整個排程。
