# 2026-08-06 研究:noRT 1.01 @ 0.3s 可行性判定與下一世代路線

**狀態**:研究定稿(未促轉任何旗標)。**分支**:`5.6-sol-reduce-time`。
**方法**:9-agent 工作流(`wf_f4a4b808-a1c`):A1 評分公式+golden 直評 / A2 殘差解剖+0.3s 操作點實測 / A3 backbone 表示解剖 / A4-A7 四路文獻掃描 → Opus 綜合 → 紅隊對抗審查 → 本定稿(已整合紅隊全部 高/中 嚴重度意見)。
**證據紀律**:內部數字可回溯 `file:line` 或 eval JSON;文獻標注證據等級(全文實讀 / abstract / 僅搜尋摘要)。原始 agent 報告與一手材料(alpha 榜 CSV、per-case median CSV、Q&A PDF、golden probe)存 session scratchpad(`wf_a1..a7.md`、`dl_*.bin`、`golden_floor_probe.*`)。
**使用者 steering(本文件排序鐵律)**:local 只有 public testcase、final 看 hidden → 不過度微調 public 參數,優先可轉移的 new method / 結構性改變;必要時可重訓 diffusion/flow-matching,但須與已判死配方有明確機制差異。

---

## 1. TL;DR(9 條)

1. **「noRT 1.01」判定(0806 晚使用者更正後修訂):條件可行,裁決繫於「修復 golden 地板」實測。** 結構要求不變:1.01 = 「逐案 HPWL/面積追平 golden(贏了不加分)**且**全場 4478 條 soft 違反壓到 ~11-22 條(golden 自己 229 條)」。**兩條新證據支持可行**:(i) 官方 QA(0618 版 A4)明文承認 golden 自帶 MIB/Boundary/Group 違規且照罰 ⇒ 修掉違規的 golden 分數 < 1.108,golden 非理論上限;(ii) 使用者情報:**別組品質分已達 ~1.01**(existence proof;架構待查證)。**但有一條硬限制**(QA A5):preplaced 位置是硬約束、不可為貼 boundary 移動 ⇒ golden 的 219 條 boundary 違規中「preplaced 自身不貼牆」者**結構性不可修**(= 0707 N2 的結論)。可修比例與修復代價(移動塊會動 HPWL,gap 是單邊罰)由 **golden 修復探針**直接量出——該數字就是 1.01 的實證裁決(行動 #-1)。擇優類機制天花板 1.0958(oracle min)之判定不變。
2. **目標的量綱很可能錯置。** 「1.01」「別組 1.08」與 alpha 榜(rank3 = 1.0197)同量級——那些是**含 runtime factor 的官方總分**,不是 noRT。以官方尺度看:**本次實測 0.3s 操作點 noRT 1.167-1.191 → 投影官方分 ≈ 0.82-0.83,已優於 alpha 第一名 0.879**(條件:hidden median 不劇烈下移、100/100 feasible;見 §2.4)。
3. **0.3s 操作點已實測,但兩條量測鏈矛盾待定案**:A2 鏈(1.1665/1.1686 @ 0.294-0.301s,對自身基線 1.186-1.190 成對 −0.020)vs fast-worker 鏈(1.1907 @ 0.329s,對基線 1.1965 差 −0.006);兩鏈 0.2s 基線本身相差 ~0.008(疑 config 漂移:A2 手抄 env 未 source `.env`)⇒ **+0.1s 的誠實增益陳述 = −0.006 ~ −0.020,方向一致、量值需 3-5 rep pinned-config 定案**(§7 行動)。runtime 軸目標已達成;品質軸殘距(對字面 1.01)仍 ~0.16-0.18。
4. **runtime 規則情報(官方一手)**:`max(0.7, R^0.3)` 在 `R ≤ 0.3046` 觸底;per-case median(公開案)給尾段免費上限 2.4-3.6s/案。我方 0.2-0.3s 檔全案觸底 ⇒ 官方分 = 0.7×noRT,**再快一分不給**;且 0.3s 檔對「median 下移」有巨大安全邊際(median ÷5 仍觸底)——**留在 ≤0.3s 是對 hidden set 最穩健的選擇**,與使用者目標一致。是否額外開「κ·median(n) 貼上限檔」(算術估官方 ~0.79-0.80,但 median 腰斬時反虧)屬時間中性鐵律修訂,**上呈使用者決策**(§5 行動 0)。
5. **「殘距 = 拓撲/表示牆」前提被三分**:(i) slicing vs non-slicing 理論面積差 0.3-1.6%、soft block 近零(BloBB 全文實讀)⇒ 「切片性」最多解釋殘距 0.008;(ii) **但**我方類別比一般 slicing 窄得多(深度≤4、分支≤4、欄數≤18、**soft 形狀=面積÷欄寬不可搜索**),這些維度 BloBB 不涵蓋;(iii) 通道 B(direct/flow→refine,非 column)實測值 −0.126,**表示多樣性確實有價**。結論:換「非切片表示」不是答案,**「拓撲固定後的聯合最優性 + soft 違反率」才是**,兩者都在精修層。
6. **首推新方法 R1:拓撲固定的座標+形狀聯合凸精修(SOCP)。** evaluator 的 HPWL 是質心 Manhattan(分段線性凸),非重疊在固定 HCG/VCG 下是線性,`w·h≥0.995A` 是 rotated-SOC,MIB/boundary/grouping/preplaced 全是線性等式 ⇒ 單一 SOCP 對所抽取的分離圖有全域最優保證,三桶(hpwl/area/V_rel)同打。機制級改變、與案件分佈無關 ⇒ **hidden set 可轉移**。兩份文獻背書(TCAD 2001 線性時間最優 sizing;arXiv 2607.21408 且其 future work 明寫要打 FloorSet)。Gate 2-3 天(§5 R1,含紅隊全部修正)。
7. **V_rel 分項解剖**:我方約束軸整體已贏 golden(violation factor 1.080/1.056 vs 1.108),3.5 檔 tail 21 案中 10 案總分贏 golden;唯一顯著劣項 = **grouping 47 條 vs golden 10 條**(存在性證明)。golden 對照背書的可收頭寸 −0.013~−0.020,上望 −0.06(未證域)。
8. **訓練線(使用者已重新授權)**:負斜率 gate 關閉的實測是在 0.2s 操作點做的,**操作點條件性**須誠實記錄(若移到更高預算檔,flow steps/DM/NREF/AL 等判死需重測);真正符合「新機制」的訓練投資排序 = oracle-prescreen 探針(半天,對 learned-ranking 整類生死判決)→ learned C* 拓撲 hint(C 不在 SA move 空間、結構上不會被退火洗掉——判死清單中「座標種子」死因的最乾淨繞開)→ data-free cost 引擎(IC/DC,賽季大注)。官方 A14「模型載入時間不計 runtime」是 ML 回歸時未被利用的規則優惠。
9. **泛化與穩健(使用者 steering + 紅隊 E3/E5)**:(i) hidden set 上任一 tail 案 infeasible ⇒ cost 10.0,總分衝擊 ~+0.3(官方 +0.2),超過本文件所有路線增益總和 ⇒ 「fallback 觸發率/收斂餘裕」量測列入立即行動;(ii) 0804-0805 以 −0.005 量級門檻促轉的五連 stack 需 3-rep 成對重驗(winner's curse 稽核);(iii) 今後促轉優先給機制級改變,全域參數微調凍結至 beta hidden 情報進場。

---

## 2. 可行性判定

### 2.1 公式與地板(A1,全部 file:line 核對)

```
Cost_i = (1 + 0.5·(max(0,hpwl_gap) + max(0,area_gap))) · exp(2·V_rel) · max(0.7, R_i^0.3)
TotalScore = Σ_i Cost_i·e^{n_i/12} / Σ_j e^{n_j/12}
```

- 常數 α=0.5 / β=2.0 / γ=0.3 / M=10(infeasible):repo 版 `scripts/iccad2026_evaluate.py:72-75`;官方版 `FloorSet/iccad2026contest/iccad2026_evaluate.py:72-75`,cost 數學逐字相同。HPWL 為**質心 Manhattan**(b2b + p2b;官方版 `:160-195`;repo 版對應段落行號不同,紅隊已校正)。
- baseline = **golden fp_sol 自己**(`_extract_baseline`),gap 下方截斷 ⇒ 贏過 golden 不加分(官方 A15 逐字確認,且「公式本賽季不改」)。
- `e^{n/12}` 是**官方 score weight**(非我方調參):n≥100 佔 82.6% 權重、n≤60 共 0.6%,有效案例數 ≈12。
- noRT 分支 runtime 項 ≡1 ⇒ **Cost_i ≥ 1.0 是結構硬地板**,實測 4 個 golden 案恰為 1.000000。

### 2.2 golden 直評(A1 新 probe,精確值)

probe 以與 `_extract_baseline` 相同的抽取把 fp_sol 餵進原生 `evaluate_solution`(`scratchpad/golden_floor_probe.py`;Lite golden 經 distinct-vertex 檢查全為精確矩形,bbox 化無損——probe JSON 的 `rectilinear_blocks` 欄位是計數 bug(把閉合頂點算入),以 distinct-vertex 直方圖 {4: 7050} 與官方 A16 為準):

| 指標 | golden | 我方 0.2s 檔 | 我方 3.5s 檔 |
|---|---|---|---|
| **加權 noRT** | **1.1079** | 1.196 | 1.115-1.118 |
| quality factor(加權) | 1.0000(定義使然) | 1.1077 | 1.0554 |
| violation factor(加權) | **1.1079** | 1.0802 | 1.0563 |
| boundary / grouping / MIB 違反 | 219 / 10 / 0 | 152 / **47** / 1 | 144 / **34** / 0 |
| 加權 packing 利用率 | 0.966 | 0.907 | — |

**修正一則長期誤傳**:memory 裡的「golden ≈ 1.0535」是 golden 經我方 decoder 重放的分;**golden 原生 = 1.1079,全部來自它自己的 soft violation**。我方 3.5 檔 tail(1.1059)已低於 golden tail(1.1074),21 個 tail 案 10 個贏 golden。

### 2.3 不可達論證(紅隊修正版)

- ~~「1.01 < golden 1.108 所以不可達」~~——**無效論證**(golden 的 qf=1.0 是同義反覆,別人追平 gap 後也拿 qf=1.0)。
- 有效論證 = 分解:達 1.01 需三項超額(現值 0.196)同比例縮到 **5.46%**;即使 V_rel 歸零,兩 gap 也要縮到現值 9.29%。時間軸已否證(17.5× 時間只買 −0.078);面積軸可挖上限 0.033(完美 packing 相對 golden 只值 −0.034,截斷後同 0);exact 通道連離線定標都不可用(n=14 / 4h / gap 20-57%,arXiv 1602.07760 全文實讀)。
- **oracle 上界**:逐案 min(我方, golden) = 1.0958 ⇒ 擇優類機制天花板明確。

### 2.4 官方尺度重讀(量綱錯置診斷)

alpha 榜(官網 CSV 一手):rank1-5 total = 0.879 / 0.955 / 1.020 / 1.028 / 1.100,avg runtime 0.96-5.32s,**無人走 0.2s 路線**。反推品質軸(誠實標注只有界):rank1 noRT ∈ **[~1.04, 1.256]**(全案觸底 ⇒ /0.7 上界;按 median 比例配時 ⇒ /0.847 估 1.04)——**無法斷言我方 1.118 是場上最佳品質,也無法斷言不是**。
我方投影(全案觸底,0.7×noRT):0.2s 檔 ≈ 0.837、**0.3s 檔 ≈ 0.82-0.83**(依 §3.1 兩鏈範圍)、3.5 檔(尾段未觸底,乘子~0.766)≈ 0.856。**條件**:hidden median 重算風險(官方 Q&A 明言 median 基於 public 案)、feasibility 100% 維持、外推非實測。

> **判定(0806 晚修訂)**:使用者裁決 KPI 維持 **noRT 1.01**、操作點鎖定 **≤0.3s**。可行性論證更新為兩段式:
> **(A) 分數存在性 —— ✅ 已實證成立**:golden 修復探針交付 **1.011256**(行動表 #-1);塑形第二階段投影 1.001-1.005。別組情報(同 evaluator/同 100 案 noRT ≈1.01,口徑已確認)為獨立佐證。R1 修復機械同場驗證:每案中位 0.08s、HPWL 代價可忽略、硬合法性零瑕疵。
> **(B) 引擎可達性** —— 存在 ≠ 我方 solver 在 0.3s 從零產出。需要:tail hpwl_gap 0.040→~0.005、area_gap 0.054→~0.005、V_rel 0.027→~0.005 **同時**,且在 0.3s 而非 3.5s。這是引擎世代問題:別組架構情報 + 修復後 golden 作為乾淨 imitation 目標(訓練線升級,§5)是兩條主通道。

### 2.5 使用者情報與 QA 補強(0806 晚)

- 「別組品質分 ≈1.01」:待確認量測口徑(同一 evaluator `total_score_no_runtime`、同 100 案)與架構(diffusion → 何種 refine/legalize?)。若屬實,直接推翻「近完美解不可工程化」的悲觀先驗,並把 §2.4 的 rank1 品質軸下界估計(~1.04)往下修。
- QA 0618 A4(一手):golden 違規「is normal and consistent with how the dataset is intended to be used」⇒ **官方設計上就預期參賽者可以比 golden 乾淨**。
- QA 0618 A5(一手):preplaced 硬 > boundary 軟,「You should never move a preplaced module to satisfy a soft constraint」⇒ 不可修違規集合的判準明確、可程式化。
- QA 0618 A6(一手):面積 1% 容差**雙側**硬檢查(與紅隊 D4 一致);「要貼牆/鄰接,正確做法是調長寬比不是加面積」——官方親自指路:**貼合問題的自由度在 aspect ratio,這正是 R1 SOCP 的變數空間**。

---

## 3. 殘差解剖(含本次實測 0.3s 操作點)

### 3.1 操作點實測(兩條量測鏈,矛盾待定案)

**A2 鏈**(`a2_op_pair.sh`,手抄 env 未 source `.env`;與工作流其他 agent 併機):

| 操作點 | rep a | rep b | avg runtime |
|---|---|---|---|
| 0.2s 基線 | 1.1855 | 1.1895 | 0.193-0.197s |
| 0.3s 初調(SCALE 1.05e-4 / MAX 1.35) | 1.1777 | 1.1843 | 0.263-0.265s |
| 0.3s 定點(OP2 調升) | **1.1665** | **1.1686** | 0.294-0.301s |

**fast-worker 鏈**(獨佔機器,基線復現 campaign 值):

| 操作點 | noRT | avg rt | tail rt |
|---|---|---|---|
| 0.2s 基線(SCALE 7e-5 / MAX 0.9) | 1.1965 | 0.2001s | 0.90s |
| run1(1.05e-4 / 1.35) | 1.2025 | 0.269s | 1.27s |
| **run2 採用(1.171e-4 / 1.505)** | **1.1907** | 0.329s | 2.74s |
| run3(1.112e-4 / 1.43) | 1.1840 | 0.271s | 1.34s |

**判讀**:兩鏈方向一致(+0.1s 改善),量值矛盾(−0.020 vs −0.006)且兩鏈 0.2s 基線互差 ~0.008 ⇒ 疑 config 漂移(A2 未 source `.env`,可能漏定案旗標)疊加家族方差;fast-worker 自己的 run1(1.2025)vs run3(1.1840)同 avg 也差 0.019,佐證此檔位 σ 偏大。**定案需同一 pinned 腳本 3-5 rep 成對**(§7)。分帶(fast-worker run2):改善集中 81-100 帶(未加權 −0.029)與 101-120 帶(−0.018)。
**附帶實錘(紅隊 C5)**:A2 op030b 首案(n=21)runtime 0.454s vs 基線 0.077s ⇒ **numba 冷 JIT 被記進第一案**——提交前必須 warm-up 護欄,否則官方機上首案 R_i 爆表。

### 3.2 分項 / 分帶(A1+A3,兩份 artifacts JSON 交叉)

- 0.2s 檔加權:hpwl_gap 0.149-0.159、area_gap 0.067-0.068、V_rel 0.034-0.037。
- 3.5s 檔 tail(權重 0.826):hpwl 0.040(=1.04×golden)、area 0.054、v 0.027 ⇒ **今天 tail 面積比線長貴**。
- 分帶:全部戰場在 n≥100 的 21 案(權重 81-83%);該帶我方(0.2s)只落後 golden 0.050;小案落後 0.20-0.35 但合計權重 <1%。
- 反事實(0.2s 檔):`hpwl→0` −0.080 / `area→0` −0.037 / `V_rel→0` −0.089 / `V_rel→min(我方,golden)` **−0.020**(背書值)/ oracle min −0.101。前 15 名加權超額案佔殘距 60%,全部 n≥97。
- **乘性套利**:β=2.0 vs α=0.5 ⇒ 每單位 V_rel 邊際代價 = 每單位 gap 的 **4 倍**;且 gap 在 baseline 處截斷(過了 golden 不再給分)。這兩點應寫進 proxy(§5 R0.7)。

---

## 4. 前提修正:表示牆的三分法(紅隊 A3/A4/A5 修正版)

1. **切片性本身便宜**:BloBB(GLSVLSI'04,全文實讀)slicing vs non-slicing 最優 dead-space 差 0.3-1.6%,soft 版 MCNC(除 apte)/GSRC 全可零 dead-space ⇒ 「換非切片表示」最多值 0.008。內部旁證:SP 窮舉 window packer(涵蓋 pinwheel)判死(1.2357);topo_search 91% hard-reject(可實現性瓶頸,非拓撲貧乏)。
2. **但我方類別窄於一般 slicing**(A3 逐行解剖):固定交替、深度≤4、Level-3 分支≤4、欄數≤18、**soft 形狀=面積÷欄寬(不可搜索)**、欄內序被 B/mid/T 分桶覆蓋、C 鏈內不可變。BloBB 的界**不涵蓋這些維度**;其中「形狀不可搜索」讓 0707「(order,shape) 不可分解」的聯合通道在 SA 層只剩一半可搜。R3 正是首次對此做參數化搜索。
3. **表示多樣性已被證明有價**:通道 B(direct/flow prior → `layout_refiner` 連續+離散精修,不經 column 表示)value −0.126(`ctrl35_dmoff`,exp 4),是全 solver 最大單一品質組件;0710「column 飽和 1.2129」正是被它打穿。`layout_refiner` 已會改拓撲(swap/insert/evict)與形狀(reshape,w·h 守恆、MIB 同步),缺的是**全域最優性**(全是局部爬山)——這就是 R1 的空格。
4. **註解修正需先歸因**(紅隊 A5):`refine/window_repack.py:4-8` 與 `slack_refiner_spec.md:21` 的「column HPWL 天花板 ~1.35×GT」在 pool 擇優輸出上已不成立(tail 1.04×),但輸出可能由通道 B 貢獻——先做零成本統計(tail 21 案 A/B 各贏幾案、通道 A 單獨 hpwl_gap),再決定改註解或加註「僅描述通道 A」。

---

## 5. 路線排序(泛化性加權後)

**排序原則(使用者 steering)**:機制級 > 參數級;hidden 可轉移性一票否決;每條附 ≤數天 gate、3-rep 成對、|Δ| ≥ 2·se 判準(σ≈0.006-0.009,單 run 不可判 <0.01)。

### 行動 0:操作點決策 —— **已裁決(0806 晚):留在 ≤0.3s**

使用者確認操作點鎖定 ≤0.3s(穩健拿 0.7 地板 + 符合目標)。κ·median 檔位存檔備查,beta hidden median 公布後可重議。以下敏感度分析保留為決策紀錄:

| 選項 | 投影官方分 | median 腰斬情境 | 判讀 |
|---|---|---|---|
| 留在 0.3s(現況) | ~0.817 | 仍全案觸底(median÷5 才開始脫底)→ **不變** | **推薦**:穩健 + 符合使用者目標 |
| κ=0.15(avg~0.70s) | ~0.807(估) | 尾段仍觸底 | 溫和選項,beta 後評估 |
| κ=0.25(avg~1.16s) | ~0.794(估) | 乘子回升 → 淨值**劣於 0.3s 檔** | 下檔風險 > 上檔收益(紅隊 C1) |

前置(10 分鐘,紅隊 E6):核對 median CSV 的 test_id ↔ `LiteTensorDataTest` block_count 對應;若不成立,κ 曲線失去立足點。**beta hidden median 公布後重錨,才是消風險的唯一動作**。另:官方 runtime 口徑(numba JIT 首編譯是否算 library loading、官方機核數)未知——提交前需 warm-up 護欄 + 冷機第一案量測(紅隊 C5)。

### R0.7:proxy 目標函數對齊(零 runtime,機制級,先做)

SA/選解 proxy(`partner/candidate_supply.py:78-101`、`column_sa_legalizer.py:1016 _key()`)若未反映 (i) V_rel 邊際 = gap 的 4 倍、(ii) gap 過 golden-baseline 即截斷,搜索就一直在買不加分的東西。稽核半天;若不符,修正 + 3-rep 成對 full-100。**hidden 可轉移(公式恆真)。**

### R1(首推新機制):拓撲固定的座標+形狀聯合凸精修(SOCP)

**機制**:取現役精修全開後的合法佈局,抽 HCG/VCG(`refine/constraint_graph.py` 已有),對 (x,y,w,h) 一次凸求解:目標 = evaluator 精確質心 Manhattan HPWL(**b2b + p2b 兩項**;p2b 是把 block 拉向正確邊界的免費線性力)+ outline ladder(×1.00/0.98/0.96 收縮判可行,對付非凸的 bbox 目標);約束 = HCG/VCG 線性非重疊、`w·h ≥ 0.995A`(rotated-SOC;**留半個容差防雙側 1% 面積檢查在浮點/取整後爆 infeasible=10.0**)、長寬比線性、MIB 同形等式(**旋轉朝向從現任解凍結**)、boundary 貼邊等式(**貼哪邊從現任解凍結**)、grouping 接觸等式 **+ y 區間正長度重疊**(角接觸不算連通)、fixed/preplaced 凍結。文獻:Young/Chu/Shen TCAD 2001(線性時間最優 sizing,全文實讀);arXiv 2607.21408(log-domain DCP legalization,「global optimality for the established topology」,future work 明寫打 FloorSet,全文實讀)。我方 `slack_solve.py`(形狀當常數)+ `aspect.py`(逐塊貪婪)恰是其啟發式退化版。

**措辭紀律**:最優性是「對**所抽取的分離圖**」而言;抽取策略(傳遞歸約/最鬆邊)列 gate ablation。

**Gate(2-3 天,離線;紅隊 D2-D8/C3/C4/E7 全部納入)**:
0. 前置 30 分鐘:clarabel/ecos 可否入提交包(`pyproject.toml` + 打包腳本);不可 → 自實作 ADMM,工程量升一級,gate 先做可行性再議。
1. 對照組 = **現役精修全開後**的 3.5 檔佈局(n≥100 的 21 案;先 5 案可行性判定再擴)。
2. 量**端到端牆鐘**(含建模;禁止 per-case cvxpy 重建,走 Clarabel/ECOS 直接 API 或預編譯 canonical form),n=120 × ladder 3 檔全跑。
3. 輸出丟回官方 `evaluate_solution` 算完整 per-case cost + feasibility(不是只算 hpwl/area)。
4. **過**:R1a(HPWL 軸)加權 tail cost −0.005 以上 @3-rep;R1b(ladder/area 軸)另計 −0.005;**100% feasibility 硬門檻**(任一案 infeasible 即判死);n=120 端到端 ≤0.3s 單核。
5. EV 一律以**淨值**表述:SOCP 增益 − 被排擠的 SA/refine 迭代損失(0.3s 檔 tail 只有 ~0.65s,序列化 carve 是真成本;若行動 0 開 κ 檔,尾段 2.4s+ 額度下此項消失)。

**判死清單交代**:不碰 prior(輸入是自家合法佈局);不拆解通道(拓撲整個保留,無跨來源拼裝);非 move/SA(一次凸解,與 Fast-SA/局部搜索/QP-hint 的「hint 餵 pipeline」死因機制不同);純 CPU。**真風險** = 現任 `layout_refiner` 爬山已近該拓撲最優 ⇒ 增益趨零——gate 正是測這個。R2 的 grouping 等式**併入同一 gate**(紅隊 D7:判準 = grouping 違反 47→≤20 且加權 cost 不惡化)。

### R1 實測裁決(0806 深夜,follow-on 全驗證)

同一 LP 修復器接我方 0.3s 輸出(成對,同一截取 run):**A+B+C 完整版 −0.0276(1.1711→1.1435)但每案中位 0.68s = 2.4× 全預算 ⇒ 不促轉**;砍 B/C(51% 時間只帶 9% 增益)。**唯一存活切片 = cheap 單發 coordinate-polish:−0.0096、97/100 案不變差、內在成本中位 24.5ms(tail 145ms,accept 改用 V10 proxy 後)、可 scipy-free(重用 `refine/slack_solve.py` 機械,boundary/grouping 強制那部分本來就無效所以正好不需要 LP)**。**→ 3-rep 成對 gate 已通過(0807 凌晨)**:Δ = −0.0096 / −0.0125 / −0.0131,mean **−0.0117 ± se 0.0011**(≫2·se)。散布疑雲解決:三 rep base 1.1711/1.1768/1.1905 橫跨歷史兩鏈全區間 ⇒ **0.02 是此檔位真實 run-to-run 變異(家族 sd ≈0.008-0.010),非 config 漂移;促轉量測一律成對設計**。**產品化已交付(0807)**:`partner/coord_polish.py` + `contest_optimizer.py` hook(`PARTNER_COORD_POLISH`,default off、off 路徑回傳同一物件、877 tests 綠)。乾淨成對 ×2:**−0.0211 @ +0.0565s avg**(補償估算後淨 ≈−0.014;時間中性控制臂進行中)。**機制發現(推翻假設):增益 100% 來自 boundary 保存列**——自由 polish 會把塊拉離牆被 gate 全數否決;`BOUNDARY=2`(只保已滿足位元)是最優模式(−0.0116→−0.0146 離線)。scipy 為唯一有效後端(numpy CSA 實測值 0,auto=無 scipy 則整個 pass 靜默不跑)。**⚠ 提交風險:官方 requirements.txt 無 scipy(也無 numba!)→ final repack 時 submission requirements 必須加 scipy(並複核 numba「官方預裝」假設)**。加速:highs-ipm + 平行項合併(t_sum 19s→4s,最壞單案 0.55s)。
**失效機理(關鍵情報)**:我方 boundary 修掉率僅 7.4%(golden 實驗 90%)——不是違規少(136+47 條更多)也不是太緊(dead space 中位 10.3% vs golden 2.9%,更鬆),而是**違規太深**:離所需牆中位 11.2% 跨距、中間隔 9 塊(golden 殘餘全是 0.5-3% 的近失)。**保序 LP 只能修近失;我方是把帶約束的塊放錯區域 = placement 問題,post-pass 救不了。**
**R6/R7 深診斷(0807 凌晨,加權校正後大幅翻案)**:
- **「深違規」是未加權統計的誤導**:計分帶(n≥100,權重 82.6%)只有 29 bits / 23 blocks,深度中位 **1.4%** 跨距(96.6% 缺口 ≤12 單位)= **與 golden 殘餘同型的近失**;「11.2% 深、隔 9 塊」來自權重 0.6% 的 n<60。R7 因此遠比想像可修。
- **P1(最高 CP,路徑覆蓋 bug)**:tag-seating 機制(`layout_refiner.py:4451 _boundary_rescue`、`:4754 _lock_compact`、`:1402 _anchor_frame_to_tags`)**只跑 direct 預測臂;column 臂勝出時版圖從未被 seat**。修補=把 column 冠軍也送進 `_boundary_rescue`(純幾何、V 嚴格下降驗收)。估 −0.010~−0.020。
- **P2(W-pin 缺失)**:`_choose_frame` 只 pin H 不 pin W ⇒ R×preplaced 違規率 60.5%(B/L 僅 5-7%)。39 條 preplaced 違規中 **27 條收框即自動消失(−0.0126)**,幾何天花板 35/39(4 條 tag 互相矛盾無解)。做法=tag 推導 W* 當 restart 臂,勿硬 pin(會觸發加寬重試反噬)。
- **P3**:`_boundary_rescue` 的 GAP=2.0 絕對門檻尺度化(→3% 短邊)可把可及率 9/29→24/29。
- **P1+P3 離線探針已跑(0807)**:函數正名 `_edge_seat`(layout_refiner.py:4451)。原樣門檻**啞彈(100 案零觸發)**;最佳修補 = GAP 係數 0.08 + (c)≤8 + **新 15 行 corner-seat pass**(雙 bit 角落 tag 對逐邊 V-單調 _commit 結構性不可見,需兩軸同步平移)= **−0.0051 @ ≤2.6ms/案、零違法**,但未達 23→≤12 判準(只到 19)。定量真兇:11/29 tagged 塊自身 preplaced(需搬牆=P2);(c2) 裁切路徑需 7-25% 面積收縮 vs 1% 容差(**永久不可行**);(c) 搬牆被 locked/clash 擋死。`_cluster_seat` 判棄(−0.0010 全在零權重小案 + 228ms 尾巴)。**教訓:內部 full_violations 驗收與官方不一致(tid 84 採納後官方反升)→ production 驗收必須用官方式 V 總和**。變體 (e) 值得 A/B(單案主導訊號,tid 97 佔一半,需 ≥2 成對 rep);真正大獎(−0.0126 連坐、−0.0228 dead)在 P2/P4。
- **P4(R6 本體,唯一有 runtime 風險)**:欄寬求解只有加寬重試、無收窄重試 ⇒ 96.8% dead 在欄內(beside_fixed 佔 39.7%)。**dead space 是結構性**:3.5 檔 dead 反而更差(0.114)⇒ 加時間買不到,與所有判死正交。dead→5% 值 −0.023。
- **P5**:grouping 計分帶只 11 條,先做 P1 看順帶效果,不獨立投資。
- **0710「frame 已 golden 級」適用範圍修訂**:aspect/H 選擇仍成立(勿再調 h_scale/0.96);**emergent W 吸收 packing 殘餘是當年未測的新軸**,不在判死範圍。
- **P 波實作交付(0807 午,三旗標 default off、off 位元組級 bit-exact 對 HEAD 驗證、896 tests 綠)**:pinned 0.3s 成對(OFF 錨定 1.1884×2 同至小數四位):`PARTNER_EDGE_SEAT_V2` **−0.0090 過 gate**(bv100 −3/−8);`PARTNER_FRAME_WPIN` **−0.0091 過 gate(0.3 檔),但檔位相依:3.5 檔中性(−0.0011,1 勝 1 負,噪音級)**——機制假說:短退火救不回壞起始 frame ⇒ 起始 frame 只在低檔位承重(與「調校即承重」定律一致);促轉建議 ≤1.5 檔啟用,跨檔需 PARTNER_FRAME_WPIN_DEBUG 儀器確認 W* 臂勝率;3.5 檔 P2 臂變異 0.0036 ≫ OFF 臂 0.0005 = portfolio 臂典型行為;`PARTNER_COL_NARROW` −0.0126 **分數過但機制 gate 不過**(dead 0.094→0.092 遠離 ≤0.07;混淆項=direct 冠軍稀釋指標+收窄迴圈吞吐稅擾動搜索軌跡;**歸因實驗前不促轉**)。三旗標全開 −0.0192(次可加,啃同一批赤字;促轉組合不可相加單旗標數字)。**量測方法學定案:跨 session 絕對值不可比(deadline-bounded SA 對機器負載敏感,OFF 錨定 1.1711→1.1884 漂移實證);一切判定限同 session 成對**。
- **⚠ 校準警告**:軟約束全滿分(ag=0,bv=0,gv=0)也只到 **1.0722**——剩餘 0.062 全在 hpwl_gap。R6+R7 整包上限 −0.035~−0.044;**通往 1.01 的最後一哩必然是 hpwl 引擎品質**(時間可買 hpwl 但我方鎖 0.3s ⇒ 需要每秒品質更高的引擎 = 別組架構線 + 修復後 golden 訓練目標)。

### R2:V_rel 針對性修復(grouping 為主)

golden 對照:boundary 我方已優(152 vs 219,無空間);**grouping 47 vs 10 = 存在性證明**。背書頭寸 −0.013~−0.020,上望 −0.06(未證域)。兩做法:(1) 併入 R1(零增量);(2) by-construction:`column_slicing.py` unit 機制(`:690-759`)已打包 cluster,問題在跨 unit 切開——`_stack_column` 分桶前加「同 cluster unit 相鄰」硬約束(3-5 天)。**hidden 可轉移(約束類型分佈不變)。**

### R3:欄寬 / banding 搜索化(表示窄類維度的首次攻擊)

`_dyn_split` 門檻(寫死 1.35)與 chunk 上限(封頂 4)從未搜索;`_width_optimize` default off 且 partner 版未移植;`h_scale` 交互未測。打 area(tail 0.054,今天最貴項)+ hpwl。Gate:0.3s 檔 + 3.5 檔,門檻 {1.0,1.35,1.8} × 上限 {4,6},3-rep 成對;**setup 成本近零但收斂成本未知**(狀態空間變大,短退火下可能反噬——必須在目標操作點量,紅隊 C6)。0707 判死的是「餵 per-block aspect hint」,不是「欄內異寬佈局」——非換皮。

### R5:內部資產移植(adaptive operator selection + racing)

vendored 雙生體已有(`column_slicing.py:1894` acc_rate×mean_impr;`:2866-2886` racing),live partner 路徑全是寫死常數(move 機率 `:2064-2092`、溫度 `:2208`)。零訓練、零文獻風險,2-3 天移植 + A/B。Gate 補紅隊 B3:**worker pool 有效 SA 迭代總數不得下降**(直接量 iters,防 AL/GPU-arm 同型的同步屏障死因)。不過 ⇒ 順帶結構性判死 Neural-SA 整支(每步 NN 推論,吞吐砍 10²-10³)。

### 訓練線(使用者重新授權;新機制類)

| 順位 | 投資 | 機制差異(vs 判死配方) | gate |
|---|---|---|---|
| T1 | **oracle-prescreen 探針**(半天) | 用 refine 後真實 cost 作弊排序,量 prescreen 類的天花板 | <0.01 ⇒ LaMPlace/MacroRank 整類判死,零訓練投資 |
| T2 | **learned C\* / 拓撲 hint**(A6-3) | C 不在 SA move 空間(`_random_move` 無增減欄 move,程式碼實證)⇒ **結構上不會被退火洗掉**,非座標種子;0.3s 檔 restart 網格 7→2-3 = 每 config 2-3× 退火步數,低檔位最值錢 | 標籤 = 既有 racing log 獲勝 config;先離線量 C 分佈熵 |
| T3 | **data-free cost 引擎**(IC/DC 2411.00003) | 換引擎非 fine-tune;唯一正攻「超 golden 品質」的文獗路線 | 賽季大注,依 0805 survey gate 條款 |
| — | 條件性記錄(紅隊 B1) | 負斜率 gate、flow steps/DM/NREF/AL 判死**皆量於 0.2s 點**;若操作點大幅上移(κ 檔),需重測後才引用 | — |
| — | 規則優惠 | 官方 A14:模型/checkpoint 載入時間不計 runtime ⇒ ML 回歸時冷載費免單 | — |

LBR(DAC'25,僅 abstract)情報:「diffusion 只當初解 + analytical 產品質」與我方通道 B 同構,與「prior 保真度斜率為負」互證——**加碼方向是強化 B 的 refine 段,不是強化 prior 段**(κ 檔開啟時把 B 的樣本數/迭代上限納入重掃)。

### 明確不建議(逐條理由,全文版見 wf_draft §5)

exact/ILP/CP-SAT 主通道(n=14/4h/gap 20-57%,連 oracle 定標都不可)|PARSAC 移植(核心機制已有:`_cfix_move`;112 核×9.3 分鐘;outline/grouping 只是軟懲罰;repo 已 archived)|冷啟 analytical 全域替換(CSF 需 ~2×10⁴ 迭代,0.1s 只買 7%;實測 numba 73µs/iter)|GPU differentiable(實測 L4 3406µs/iter,比單 CPU 核慢 46×;n²=1.4e4 填不滿 GPU)|learned prescreen 直投(先過 T1 探針)|AutoDMP 式 per-design BO|T-REX/reward-extrapolation(負斜率 gate)|windowed MILP 再投(exact window 上限 ~8-9 blocks;R1 通了再議)|R4 learned 預算重分配(官方乘子是 per-case 地板,跨案重分配在官方尺度無意義——紅隊 B2;noRT 尺度下先跑 2 小時離線 oracle,≥0.010 才議)。

---

## 6. 泛化與穩健(hidden-set 紀律)

1. **infeasible 尾風險(紅隊 E3,單項量級最大)**:tail 一案 infeasible ⇒ cost 10.0,noRT +~0.3。行動:量 0.2s/0.3s 檔的 row-fallback 觸發率、最小收斂餘裕、以及「最難案再難 20%」的壓力測試;robustness 是留在 0.3s 檔的第二理由。
2. **winner's curse 稽核(紅隊 E5)**:promoted 五連 stack vs 促轉前基線,3-rep 成對重測,記錄真實總增益 ± se。0804-0805 的 −0.005 級門檻在 σ 0.006-0.009 下有過擬合 public 的風險。
3. **促轉紀律更新**:機制級改變(R0.7/R1/R2/R5)優先;全域參數微調(SCALE/steps/NREF 家族)凍結,僅在 beta hidden 情報進場後重錨;所有 gate 3-rep 成對 + 共用種子 + |Δ|≥2·se。

---

## 7. 立即行動清單(本週)

| # | 動作 | 成本 | 判準/產出 |
|---|---|---|---|
| -1 | ~~golden 修復地板探針~~ **✅ 已完成(0806 晚)**:**修復後 golden = 1.011256**(shape 凍結保序聯立 LP + 牆邊反轉 + grouping 橋接;100/100 feasible、嚴格稽核零瑕疵;197/219 boundary + 7/10 grouping 修掉,HPWL 代價 ≈0.00014 可忽略,87 案修完 HPWL 反而更短)。**存在性成立**。殘餘 25 條:13 條 preplaced 硬鎖(74% 殘距)、9 條拓撲卡死、3 條 grouping;case 89/99/88 佔 68%。塑形第二階段(交替 LP,固定 h 解 (x,w) 為真 LP)投影 1.001-1.005,半天。附帶:golden 非其拓撲下的 HPWL 最優;`vsnap.py` 是本機制的逐塊貪婪版,聯立 LP 一次修 189/219。工具:SESSION_SCRATCH/gr_lib.py + gr_repair3.py。**follow-on 進行中:同一修復器接我方 0.3s 輸出全驗證(= R1-stage-A 促轉裁決)** | 探針:每案中位 0.08s |
| 0 | **0.3s 操作點定案**:單一 pinned 腳本(source `.env` + 完整旗標)3-5 rep 成對 0.2s vs 0.33s | 半天(機器時間) | 增益範圍 −0.006~−0.020 收斂到 ±2·se |
| 1 | median CSV ↔ block_count 對應核對(E6) | 10 分 | 不成立 ⇒ κ 曲線作廢 |
| 2 | 通道 A/B tail 歸因統計(A5) | 零(讀既有 JSON) | 決定 1.35× 註解怎麼改 |
| 3 | R0.7 proxy 對齊稽核 | 半天 | 不符 ⇒ 修正 + 3-rep |
| 4 | R1 依賴入包確認 → gate probe(併 R2 grouping 等式) | 30 分 + 2-3 天 | §5 R1 判準 |
| 5 | T1 oracle-prescreen 探針 | 半天 | <0.01 ⇒ 整類判死 |
| 6 | infeasible 尾風險量測(E3) | 半天 | fallback 率 + 收斂餘裕報告 |
| 7 | promoted stack 3-rep 重驗(E5) | 1 天(機器時間) | 真實增益 ± se |
| 8 | 文件/memory 更正:golden 1.0535=decoder 重放(原生 1.1079);1.35× 註解待 #2;probe 頂點計數 bug | 1 小時 | — |
| 9 | 行動 0 決策包呈使用者(κ 選項 × median 敏感度表) | 已含本文件 | 使用者裁決 |

---

## 8. 文獻總表(節選;完整表與證據等級見 wf_draft §7)

全文實讀:BloBB(GLSVLSI'04)· ISPD'05 表示重要性 · Young/Chu/Shen TCAD 2001 · arXiv 2607.21408(凸 legalization,將打 FloorSet)· arXiv 1602.07760(FLP exact 上限)· PARSAC 2405.05495 · FloorSet 2405.05480(**論文無任何數值 baseline;Lite golden 的 nets 是放置後反推採樣 ⇒ golden HPWL「近最佳」是構造性的**)· CSF 2504.03796 · PeF · Per-RMAP 2304.06698。
abstract/摘要級:LBR DAC'25(IEEE 牆)· ICCAD'24 Rectilinear Soft Modules(10.1145/3676536.3676818,**約束覆蓋最接近我方、自稱贏 contest 冠軍隊——下輪唯一任務=取得全文**)· RulePlanner 2601.22476 · Neural SA 2203.02201 · MacroRank / LaMPlace · Damani 2410.04707 · GrandPlan ISPD'26。
官方一手(已下載):alpha 榜 CSV、07/21 per-case median CSV、08/05 Q&A(A12/A14/A15/A16)。
FloorSet 引用面窮盡(S2+OpenAlex 聯集 9 篇):**至 2026-08 無任何公開論文報告過 FloorSet-Lite 100 案 evaluator 分數**——外部唯一校準 = alpha 榜。

## 9. 未解決不確定性

官方分投影全部是算術外推(0.7×noRT),非官方重算|hidden median 重算方向未知(公開 median 基於 public 案,官方 Q&A 原文)|test_id↔block_count 未核對(行動 #1)|R1 SOCP 端到端牆鐘無數字(gate 第一量測)|MIB 旋轉/hidden 集 rectilinear 同構性未知(Lite 實測全矩形)|官方機核數與 runtime 口徑未知(JIT 首編譯歸屬——**本次已實錘會被記進首案**,見 §3.1)|ICCAD'24 全文未取得|**0.3s 點兩量測鏈矛盾未定案**(−0.006~−0.020,行動 #0)。
