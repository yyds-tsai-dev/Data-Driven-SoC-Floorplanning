# 2026-07-07 研究合流:order-decoder 路線圖(降 total_score_no_runtime)

**目標**:production 1.2298 → 逼近 decoder 理論上限 ~1.05。
**方法**:deep-research workflow(5 angles / 18 sources / 88 claims 抽取)+ deep-reasoner(Opus)實際讀碼分析,交叉合流。
**查證狀態註記**:workflow 的 3 票對抗查證階段因 5-hour usage limit 全數中斷(25 claims 全標 UNVERIFIED)。其中 FAST-SP、Fast-SA、TILOS、blackbox-solver differentiation 的內容與模型自身知識一致(高信心);ChipDiffusion 的 arXiv ID/標題/方法已另行 WebFetch 覆核;MDPI RL-over-SP 與 GPU-SA 僅 abstract 級證據。**所有期望 delta 都是推論,決定性證據一律以本地 probe 為準。**

---

## 0. 既有量測基準(校準用,非本次研究產物)

| 量 | 值 |
|---|---|
| production(COLUMN_BACKBONE+SLACK_REFINE+FAST_EVAL+WINDOW_REPACK) | **1.2298**,100/100 feasible,~4.9s/case |
| GT 直接評分 | 1.1079(GT 自帶 boundary 違規) |
| golden-order exact decoder(order→longest-path compaction→exact-area→wall-snap) | **1.0535** |
| column 表示 HPWL 天花板 | ~1.35× GT |
| GNN 0514 hints 經 v2 修復管線 | 2.92(fill 56.5% vs golden 97.1%) |

殘餘 gap 幾乎全是 HPWL 拓樸品質;violation 線已收案(v_rel 0.0335 優於 GT 0.053)。

**讀碼修正兩則(影響所有 gating 敘述)**:
1. production path 在 `src/floorset_arch/optimizer.py:176` 直接短路進 column backbone,**完全繞過 budget_layer / quality_portfolio / v10_proxy**(它們只活在 legacy path)。新槓桿的 runtime gating 必須在 `column_backbone.py` / `refine/` 自建 reserve(現有 `topo_reserve` L199-217 是樣板)。
2. `_time_budget`(`column_backbone.py:28-32`)= `clamp(0.06·e^{n/20}, 0.8, 24)`,純 n 函數。按可重用統計(hpwl_gap、net density)把預算從低 gap 大 case 挪到高 gap tail 是合法重配(非 test_id hard-code)。

---

## 1. 排序後的槓桿

| # | 槓桿 | 期望 delta(對 1.2298) | 信心 | 前置條件 |
|---|---|---|---|---|
| ~~**D1**~~ | ~~decoder-as-polish~~ **☠ 0707 MEASURED KILL**(deploy delta −0.0000,2勝98敗,fallback 33/100) | ~~-0.01 ~ -0.04~~ | — | 見 [docs/experiments/2026-07-07-d1-decoder-polish-probe.md](../../docs/experiments/2026-07-07-d1-decoder-polish-probe.md) |
| ~~**A**~~ | ~~order-space local search~~ **☠ 0707 連帶 KILL**(先決 D1≥−0.01 失敗;realizer 有 violation 稅+33% 不合法率) | ~~-0.03 ~ -0.10~~ | — | 同上 |
| ~~**B**~~ | ~~SP window packer~~ **☠ 0707 MEASURED KILL**(SP 邊際 0.6/221.7 HPWL ≈ 1e-5 total-equiv;9/252 accepted;拓樸維度無礦,flex 才是載體;證據 docs/experiments/2026-07-07-b-sp-window-packer.md) | ~~-0.003 ~ -0.008~~ | — | 程式碼保留、旗標 OFF |
| **C** | ML pair-cost head + blackbox-solver differentiation | -0.005 ~ -0.02 | 低 | scale probe 判定(A 死後 consumer 改為 decoder-probe 路徑自身) |
| 停車 | ChipDiffusion 式 legality-guided sampling | — | — | scale probe 存活且 diffusion 復活 |

**核心洞見(事後驗證):D1 是 A 的零搜尋特例,一次 probe 同時判了兩者死刑——probe 成本 ~450s,避免了整套 order-SA 的白做。**
**Post-mortem 要點:HPWL/area 在 decode 往返近乎無損(hpwl delta +0.0025)——死因是 soft-violation 稅(v_rel 翻倍,decoder constraint pass 弱於 refine 疊層)+ 33% hard-legal 崩潰。golden→1.0535 仍成立,ML 學 golden-like order 的動機反而更強。**

---

## 2. Lever D1 — decoder-as-polish(先做)

**機制**(file:line 皆 deep-reasoner 實讀):`build_order_dags`(`scripts/probes/gen_decoder_probe.py:229`)對 hint 的唯一依賴是 centroid 相對位置——不需要 golden。把 production 的合法 layout 當 hint 餵入 = 對它自己的拓樸做:
1. `longest_path_coords`(L291)ASAP compaction——解除 column 量化、擠掉欄間 dead space(column SA 的 `_cost` 座標受欄邊界量化,這是無損最緊 packing);
2. `boundary_snap`(L371)修 SA 結構性修不掉的牆違規——正是 golden→1.0535 打敗 GT 1.1079 的來源;
3. production order 已被 SA 收斂,不會像 GNN order 那樣 fill collapse。

**驗證(近乎免費)**:`gen_decoder_probe.py` 加一個 `--hints production` source(餵 backbone 輸出),跑 100 case。不需改 src。

**上線實作**(若 probe 為正):把 `build_order_dags` / `longest_path_coords` / `decode_shapes` / `boundary_snap` 從 probe 提升為 `src/floorset_arch/refine/decoder_polish.py`;接在 `column_backbone.py:268` `refine_layout` 之後;gate = `refine/guards.py` 的 evaluator-faithful guard + strict-better(hard_legal 且 score 嚴格較低才採用,失敗回傳原圖)。

**Kill criterion**:probe weighted total ≥ 1.228(< 0.002 改善,在 SA 噪音帶 ±0.006 內)→ kill D1+A。若 `legal_fallback` > 5/100 → production order 有 pin 不相容,先接 v2 的 `repair_pin_order`(`gen_decoder_probe_v2.py:262`)重測一次再判。

---

## 3. Lever A — order-space local search(最大上限)

**與已判死 M2 的本質差異**:M2 的投影內層求解器(`project_axis`,dims-frozen)不重推座標,DAG edit 推不動 incumbents;exact decoder 從 order **重新推導全部座標**,move 真的生效。

**設計**(文獻 + 讀碼合流):
- **Seed**:production layout 抽出的 order(= D1 的 order)。起點 ≤ production,搜尋只會改善。
- **Move set**:相鄰軸 swap、軸翻轉(x↔y,修 `build_order_dags` L256 tie-break 選錯軸)、block reinsertion。與 RL-over-SP 文獻的 5-move 集合互相印證(swap in Γ+ / swap in Γ− / cross-swap / reposition / rotate)。
- **內圈評估(關鍵升級,來自文獻)**:sequence-pair → 座標可用 **weighted-LCS** 計算(Tang-Tian-Wong TCAD 2001):簡單 O(n²) 版常數極小,2001 硬體 n=128 就有 60× 於 constraint-graph longest-path、~10⁶ evals/min。deep-reasoner 的保守估計(full decode 10-30ms → 2-3s 買 100-300 evals)可上修 1-2 個量級:**內圈用 LCS proxy(座標+HPWL O(E)),faithful decoder(pin floors/exact-area/MIB shared aspect/boundary snap)只對 improving candidates 跑**。proxy-faithful 落差風險由二層評估 + 端到端 kill 準則吸收。
- **Schedule**:Fast-SA 三段式(高溫隨機 → pseudo-greedy T→0 → reheat hill-climb;Chen-Chang ISPD'05,GSRC n100 5.5s vs 古典 SA 127s)。
- **平行**:go-with-the-winners 週期性克隆勝者(TILOS MacroPlacement 2025-03 的 SA 升級 pattern),複用 `column_slicing._parallel_solve` 基建。
- **Pin 相容**:preplaced pins 在 `longest_path_coords` 是硬 floor(L326-353);move 違反 pin 相容時用 `repair_pin_order` 投影修復。
- **目標 case**:用可重用統計 gate(n≥60、net density 高、D1 解碼後 hpwl_gap 仍 > 閾值),預算從 `topo_reserve` 樣板挖 2-3s。
- **佐證**:TILOS/UCSD 獨立評測(arXiv 2302.11014 + CACM)——SA 幾乎總是以更少算力打贏 Circuit Training 的座標 RL,支持「古典 order search 優先於座標 ML」的排序。

**Kill criterion**:先決 D1 ≥ -0.01。A 本身:同 runtime paired log(JSONL)full-100 改善 < 0.005 → kill(提防 SA budget 被挖傷,前車之鑑 avg_rt 4.46→4.00)。

---

## 4. Lever B — M5 v3:SP window packer(估殘餘)

**[0707 實作已落地(tests 31/31 綠),全量量測 pending]**:`src/floorset_arch/refine/sp_pack.py`(k≤5 全 (k!)² SP 窮舉,沿用進場 shapes 不 resize,numpy 向量化 longest-path + ASAP/ALAP×兩軸 4 種 justification,top-N 去重後走 `_consider`)。閘 `FLOORSET_WINDOW_SP_PACK`(預設 OFF)、`FLOORSET_WINDOW_SP_K`(5,硬上限 6)、`FLOORSET_WINDOW_SP_TOPN`(24)。per-window JSONL 新欄位 `packer` 與 `classic_best_hpwl`(counterfactual)→ 單一 SP-on run 自足歸因 SP 邊際增益;stage detail 加 `n_sp_accepted`。測試:tests/test_sp_pack.py。

**no_repack root cause(讀碼)**:`_repack_window` 的 slicing packer(`window_repack.py:569`)對近滿窗(~100% fill 的 column sub-layout)自由度只剩 aspect flex + cut 方向,擠不贏 SA 已收斂的 baseline(L838-839 strict-beat 條件)→ 575 窗 no_repack 是**表示能力**問題,不是 bug。

**升級**:k≤5 窗做 SP 全枚舉((5!)² = 14,400 × LCS eval ≈ ms 級/窗);k=6(518k)用抽樣或 numpy 批次。packer 複用 D1 提升到 src 的 longest-path/LCS 模組。gate chain 不動(L1085-1101 已 evaluator-faithful)。

**CP-SAT 後備**:模型只要 ~30 行(cpsat-primer packing recipe:整數 bottom-left 變數 + `add_no_overlap_2d`),k≤10 遠低於 exact 2D packing 硬度前緣(~20 items 才開始有 open instances,survey arXiv 2004.12619);但 recipe 是純 feasibility(無 objective,預設 900s harness),要加 HPWL objective + 硬 time cap,且 **ortools 依賴有 PyInstaller 打包風險 → 純 Python SP 優先,CP-SAT 只在 SP 枚舉不夠時**。

**注意**:B 與 D1/A 高度重疊(D1 是 global 版的同種 compaction)。**若 D1 上線,B 只估殘餘**;paired log 改善 < 0.003 → kill。

---

## 5. Lever C — pair-cost head + blackbox-solver differentiation(ML 唯一活路)

**規避 0706 kill 的方式**:不再「分類模仿幾何標籤」(已證明打不贏幾何抽取 0.858 vs 0.897)。改為:
- NN 預測 pairwise costs **w** → 結構化 order solver(argmin **w**·φ(order),強制全域一致/雙軸 acyclic)→ 輸出 order;
- Loss = **Hamming distance vs golden order**(監督在離散結構上,labels 從 golden 過 evaluator 抽取——正是 kill 註記要求的換標籤方向);
- Backward = **一次額外 solver call**(perturbed weights w' = ŵ + λ·dL/dy,梯度 = −(1/λ)[ŷ − y_λ];Vlastelica et al., ICLR 2020, arXiv 1912.02175);訓練每 step 只要 2× solver runtime,n≤120 可行;實務 λ ≈ 10-20(論文反直覺發現)。
- **凍結 encoder/anchor**(0706 教訓:phase-2 聯合微調漂移 anchor 通道)。
- 消費端:`--pair-blend` + `build_order_dags_blend` 已在 probe v2(高信心 disagreement 才 override)。

**上限誠實估**:幾何抽取已對 98.7% pairs,錯的 1.3% 是低訊噪難 pair;且 A 的軸翻轉 move 也能修同一批——C 的增量只在 A 搜不到的 case。**Promotion gate**:decoder-probe area_gap < +0.3、fill > 85%、`head BEATS geom` > 0、full-100 > 0.005。

**排程**:等 scale probe 判定(ep5-6 規則)。scale 死 → C 是唯一 ML 牌,且要等 A 的 consumer;A 也死 → ML 全下架,資源回 Track A/B。

---

## 6. 停車場

- **ChipDiffusion 式 legality-guided sampling**(arXiv 2407.12282, "Chip Placement with Diffusion Models", Lee/Nguyen/Elzeiny/Deng/Abbeel/Wawrzynek + github.com/vint-1/chipdiffusion):quadratic pairwise-overlap potential + HPWL potential 經 backwards universal guidance 進 sampling loop,合法性不靠外部 legalizer。E4 屍檢顯示 v11 diffusion 的 3.39 全死在 legalization(raw prior 1.09×HPWL / 21% overlap)——此文獻正攻這點,落點 `src/floorset_arch/diffusion/sampling.py`。**只在 scale probe 存活且 diffusion 路線復活時啟用**(zero-shot 數字 2.49e5 vs MaskPlace 8.72e5 未查證)。
- **Within-chain 併發 move**(GPU-SA 文獻,abstract 級):單鏈上多 move 併發評估,與 parallel-restart / PARSAC 獨立 worker 是不同平行軸;我們純 CPU,只取「move 批次向量化」想法,優先級低。
- **Per-case budget 重配**:~~作為 A/D1 的 budget policy 併入~~ **☠ 0707 規則層面 INVALID**——官方 RuntimeFactor 是 per-case 對全體參賽隊伍中位數(PDF p.5 腳註3 + evaluator L925-931),每個 case 獨立比較、慢懲罰不封頂;「挪時間給 tail、中位數不動」不成立。本地目標=total_score_no_runtime,runtime 屬提交期風險管理。見 CONTEXT.md「Official RuntimeFactor」詞條。

---

## 7. 執行順序決策樹

```
1. D1 probe:gen_decoder_probe.py --hints production(100 case,近乎免費)
   ├─ total < 1.228           → D1 上線(refine/decoder_polish.py)→ 開 A
   │    └─ A paired <0.005    → kill A;B 估殘餘
   ├─ 1.228–1.2298(噪音帶)  → 看 fallback:>5/100 → 接 repair_pin_order 重測一次再判
   └─ ≥ 1.2298                → kill D1+A → B(SP packer)      ◄◄ 0707 實測走此分支
2. B:在 D1 判定後只估殘餘(SP packer vs slicing packer paired log)   ◄◄ 現任下一棒
3. C:等 scale probe 判定;blackbox-diff 配方如上(consumer 改 decoder-probe 路徑)
```

**0707 判定紀錄**:decoded 5.3367 vs baseline(同 run)1.2781,portfolio(min) −0.0000;
repair_pin_order 分支不啟用(非 fallback 子集平均仍 +0.168 cost,方向不可翻轉)。
完整證據:docs/experiments/2026-07-07-d1-decoder-polish-probe.md。

---

## 8. B 之後的下一波(0707 晚,deep-reasoner 讀碼確證 + 官方 PDF 規則合流)

| 排名 | 槓桿 | 期望 delta(no-rt) | 成本 | 判定依據 |
|---|---|---|---|---|
| ~~**1**~~ | ~~N1 面積 slack 全域縮放~~ **☠ 0707 MEASURED KILL**(area −0.0139 ✓ 但 n≥100 hpwl **+0.0212 系統性**(5/21 勝,p≈0.013);加權配對 +0.0052 反向。疊層是 exact-area 調校的 local optimum,全域擾動 de-tune。旗標休眠。證據 docs/experiments/2026-07-07-n1-area-scale-n2-tail-violations.md) | ~~−0.004 ~ −0.008~~ | 已量測 | control 重確認基準 **1.2297** |
| ~~**2**~~ | ~~N2 tail 違規法醫~~ **☠ 0707 KILL**(golden 對照:case99 我5/金4、case89 我6/金7、case88 我7/金7 → 結構性違規,golden 同帶;可收僅 ~1 個違規 ~−0.003。0705 判定獲 case 級確認) | ~~−0.01 ~ −0.03~~ | 3 張 PNG | 同上文件 |
| ~~3~~ | ~~N3 batched-move SA~~ **☑ 0707 收案**(profile 否決 batched 設計:熱點是幾何重推非 numpy 部分,天花板 1.2×;有界微優化落地 bit-exact 30/30 綠:mid +18.7% 但 **tail +0.8%=權重所在紋絲不動**,total <0.001 低於量測地板;深度重寫被打包風險+「調校即承重」否決。證據 docs/experiments/2026-07-07-n3-sa-throughput-scale-pt2.md) | ~~−0.005 ~ −0.008~~ → <0.001 | 微優化保留(利 RuntimeFactor) |
| ☠ | Q4 restart 結構多樣性 | ≤ −0.002 | — | **已內建**:`_parallel_solve`(column_slicing.py:2759-2768)12 configs 已變動 column count C0±2/orientation/seed/v_weight;僅剩 width-profile 擾動,毗鄰已死 width-opt。kill |
| ☠ | Q2 tail 預算重配 | — | — | **規則層面 INVALID**(PDF p.5 腳註3:per-case 對全體提交中位數;user 指正確認)。見 CONTEXT.md |
| ☠ | Q5 cluster edge-touch bonus | 0 | — | scoring.py:66 的 −10× bonus 屬 legacy 路徑,production 不呼叫;evaluator 只罰不連通(連通分量−1),無正 bonus 可捕 |
| 長線 | 路線 C(ML pair-cost head + blackbox diff) | 通往 0.18 headroom 的唯一路 | — | 等 scale-probe 判定;與 N1-N3 正交。**N1-N3 合計上限 ~−0.015,碰不到 order-quality 天花板** |

**Do-first:N1(面積 slack)**,一次 eval_total 定生死;N2 診斷平行做(讀 PNG/診斷表,零實作)。

## 9. 引用清單(標注驗證狀態)

| 來源 | 用途 | 驗證狀態 |
|---|---|---|
| Tang & Wong, "FAST-SP", ASP-DAC 2001 | SP→placement O(n log log n) | 與模型知識一致;未逐字查證 |
| Tang, Tian & Wong, "Fast evaluation of sequence pair … LCS", TCAD 2001 | 內圈 LCS proxy;60×@n=128、~10⁶ evals/min(2001 硬體) | 與模型知識一致;數字未逐字查證 |
| Chen & Chang, "Modern floorplanning based on B*-tree and Fast-SA", ISPD'05/TCAD'06 | 三段式 schedule;fixed-outline WL 11-20s@33-49 blocks | 與模型知識一致 |
| TILOS-AI-Institute/MacroPlacement + arXiv 2302.11014 + CACM 評論 | SA > Circuit Training;go-with-the-winners(2025-03) | 高信心(repo 公開紀錄) |
| Vlastelica et al., arXiv 1912.02175(ICLR 2020) | Lever C 的可微分 blackbox solver 配方 | 與模型知識一致,高信心 |
| Iori et al., arXiv 2004.12619(exact 2D packing survey) | exact 硬度前緣 ~20 items;CP 是近年最有效方向 | 高信心 |
| d-krupke/cpsat-primer packing recipe | CP-SAT 窗模型 ~30 行;feasibility-only 注意 | repo 公開程式碼 |
| ktnr/BinPacking2D | CP-SAT 當 feasibility oracle 的 B&C 分解 | repo 公開程式碼 |
| MDPI Appl. Sci. 14(7):2905(2024) | RL-over-SP 5-move 集合;絕對品質弱(DS 11.55%) | **未查證,abstract 級** |
| "Optimizing SA on GPU … IC floorplanning"(ResearchGate 220729765) | within-chain 併發 6-160× | **未查證,abstract 級** |
| arXiv 2407.12282 "Chip Placement with Diffusion Models" + vint-1/chipdiffusion | 停車場:legality-guided sampling | **標題/作者/方法已 WebFetch 覆核**;數字未查證 |
| 延伸閱讀(抓取未消費):arXiv 2407.15026、arXiv 2604.23658、Pu et al. ISPD 2024(yibolin.com)、limbo018/DREAMPlace | — | 未讀 |

## 9. 未驗證假設(誠實標記,全部以本地量測為準)

1. D1 的 -0.01~-0.04 是推論——production order 經 `build_order_dags` 抽取的品質是否夠好,唯一決定性證據是 `--hints production` probe。
2. A 的吞吐上修假設 LCS proxy 與 faithful decoder 的排序一致性夠高;若 proxy-gap 大,回退到 full-decode 內圈(100-300 evals,上限較低)。
3. 文獻數字多為 MCNC/GSRC(area 或無 FloorSet 式硬約束),只支持 pattern 可行性,不保證我們 benchmark 上的絕對增量。
