# 0.3s 檔 hpwl 赤字解剖與引擎線開案設計備忘錄

日期 2026-08-07|分支 `5.6-sol-reduce-time` @ 59686c4|純唯讀分析(deep-reasoner),未改任何檔、未跑 solver。A 級數字可用 session scratchpad `dr/` 下腳本重算(`anat.py` `bands.py` `cf.py` `percase.py` `var.py` `vardecomp.py` `harvsplit.py` `marginal.py` `frame.py` `framecorr.py` `fslice.py` `partial.py`)。

## 0. 決策摘要

1. **「hpwl 是剩餘最大桶」只對了一半,必須修訂。** 出貨 0.3s 操作點加權赤字 0.169 = hpwl **0.070** + area 0.033 + violations **0.067**(分數單位)。hpwl 只險勝 violations;**110-120 帶(60% 權重)violations(0.042)遠大於 hpwl(0.023)+area(0.015)**。hpwl 是主戰場是因為它是唯一對時間有彈性的桶(0.3→3.5 的 −0.047 中佔 83%),不是因為最大。
2. **時間軸已耗盡,(c) 參數重掃降級。** +1.52s(6.3×)只買 0.047 = 每秒均時 0.031 分;分帶邊際顯示 tau=12 曲線形狀已接近最優;21-59 帶邊際恰為 0.000 但只釋出 0.023s。
3. **最大未開採槓桿 = 零加時的運氣消除。** 同 config 4 reps 的 per-case best-of-4 oracle = **−0.0158~−0.0202**(≈3.5 檔全部時間優勢的 34-43%,成本零秒);3.5 檔同指標塌到 0.0048 ⇒ 短退火專屬病理。
4. **hpwl 赤字有低維可學主因:frame aspect。** n≥100 帶 `|log(AR_ours/AR_golden)|` 對 hpwl_gap Spearman **ρ=0.74**(控制 area_gap 偏相關仍 **0.62**);|logAR|≥0.05 佔 45% 權重扛 **80% hpwl 赤字(0.056/0.070)**;3.5 檔 ρ 掉到 0.355 = 長退火自己修好 frame(與 FRAME_WPIN 檔位相依互為佐證)。
5. **開案順序改寫:T1 oracle-prescreen(半天,判死/放行整類)→ G3 兩行選擇器修正 → T2′ 學 frame 標量(非 learned C*)→ (d) `_axis_pass(hold=False)` kernel 化 → T3 修復後 golden imitation(條件性大注)**。(c) 最低優先。

## 1. 赤字解剖(A 級,eval JSON 重算,total 對到小數 6 位)

**口徑警告**:先前「0.3s 檔 hpwl 加權 ~0.10-0.12」是原始 gap 加權,非分數單位;分數單位 = `Σ w·0.5·h·VF` 等,差約 0.55×。**gate 判準一律用分數單位。**

分帶(0.3s 檔、P 波全開、2 reps):

| 帶 | 權重 | hpwl | area | viol | 赤字 |
|---|---|---|---|---|---|
| 21-59 | 0.006 | 0.0011 | 0.0004 | 0.0005 | 0.0019 |
| 60-89 | 0.069 | 0.0154 | 0.0041 | 0.0055 | 0.0250 |
| 90-99 | 0.098 | 0.0176 | 0.0041 | 0.0080 | 0.0297 |
| 100-109 | 0.226 | 0.0129 | 0.0088 | 0.0107 | 0.0324 |
| **110-120** | **0.600** | 0.0228 | 0.0152 | **0.0420** | **0.0801** |
| 合計 | 1.000 | **0.0699** | 0.0325 | **0.0667** | **0.1691** |

反事實:viol→0 = 1.0964;hpwl→0 = 1.0993;area→0 = 1.1366;hpwl+viol→0 = 1.0305(3.5 檔:1.0562/1.0912/1.0933/1.0271)。**更正 memory:「軟約束全滿分 1.0722」應為 0.3s 檔 1.0964 / 3.5 檔 1.0562。**

時間反事實(0.3→3.5):總 −0.047,hpwl −0.0389(83%)。逐案頭部集中:tid 96/97/74/80/76/99 佔 90%。真拓撲牆:tid 79/81/75(時間救不回)。反向案(98/95/87/92 時間多反而 hpwl 差)= 選擇器 trade-off 證據。

通道歸因(B 級,n=95 斷點推斷):n≥95(權重 0.886)4-reps 全距 0.027、harvest 0.0236;n<95 僅 0.0049。n≥100 有 14/21 案 4 reps 逐位相同,變異集中 tid 86/89/99/77/91 = 冠軍兩臂切換簽名。

**選擇器兩個可疑點(A 級,`_parallel_solve` :3731-3755)**:(i) 內部 score 的 `hp_ref` 用 pool 最小 hpwl,evaluator 用 golden ⇒ 高 gap 案(0.4-0.7)hpwl 懲罰被壓縮 40-70%,高估 violations;(ii) `0.985` 換臂死區 ≈ 絕對 0.015-0.02,大於多數 A/B 增益。兩者零 runtime、可獨立 A/B。

## 2. 機制排序

### (a) racing/adaptive — EV 上界 0.016,實得估 0.004-0.008;**必須先過 T1**
繫到 best-of-4 harvest。兩種互斥機制:M1 選擇誤差(候選在池、proxy 排錯 → racing/選擇器可拿)vs M2 深度運氣(候選沒被生出 → 只有 kernel 能買)。反證:POOL46/擴列判死支持 M1,但 T1 是唯一區分實驗。gate:racing 需在 T1 的 M1 份額上實現 ≥50%;M1<0.005 ⇒ racing 整條判死。

### (b) T2′ = 學 frame 標量(W*/AR*),取代 learned C*
**原 T2 論證不成立**:`_POOL_SIZE=24`,每 config 各佔一 worker 共用同一 deadline,收窄 C 網格不延長退火,只換 order statistics(已被 POOL46 判死打臉)。
T2′:標籤免費(golden bbox 不受 soft violation 汙染,1M 樣本零修復成本);消費管道已在生產碼(`_w_star_from_tags`→`w_star`→`H=A/(0.96·W*)`),學到的 W* 接同一插槽、每案可出(現行只在 R-tag preplaced 觸發);繞開全部死因(非 imitation 座標、非 prior 保真度、1 純量推理 ~0、A14 載入免費、下檔封閉=只是多一條輸掉 true-cost 的臂);EV 上界 0.045、保守 0.010-0.020。
**G-T2′-0(oracle probe,0.5 天零訓練)**:golden W_g 寫檔餵 `_ws_map`,3 對成對。≤−0.010 且 hpwl 佔 ≥70% ⇒ 放行訓練;>−0.005 ⇒ frame 線全死。**本備忘錄投報比最高的單一實驗。**
G-T2′-1(訓練 2 天):小 MLP/set-transformer,輸入=面積分佈+pin 方向直方圖+約束密度+n,輸出 log AR*;判準 hold-out |logAR| 中位數 < 現行的 50%。

### (c) NREF/flow-slot/phase-B 重掃 — 暫緩(邊際耗盡 + hidden 轉移風險;若做必須 frontier 括號 + 判準 ≥−0.010)

### (d) refine 段 kernel 化 — 標的已定位:`_axis_pass(hold=False)`
`refine_numeric_kernel.py` Scope 自述:只移植了 hold=True(合法化半邊);**HPWL median sweep(品質半邊)100% Python**,而 `_Refiner.run()` 主迴圈(:3946/4043/4056)正是它。`_axis_constraints`(53% rung)已寫好;只缺 njit `_median_shift`/`_wmedian`。**`_wmedian` 對 tie 置換不變**(argsort 置換只在相等值區塊內,回傳值唯一;殘餘風險=cumsum 浮點序 ULP → 契約改「100 案+單測逐位相同」實測制,同 sa_numeric_kernel 規格)。
G-d(2 天):先儀器量 hold=False 佔 rung 牆鐘(≥25% 續、<15% 收案)→ 移植+`==` 測試 → frontier 括號 ×3,判準 ≤−0.005。kernel 收益隨預算遞減對我們有利(目標檔正是 0.3s)。

### (e) T3 修復後 golden imitation — 條件性大注,**先不投**
成本非障礙(1M 修復 ≈ 55 CPU-hr ≈ 48 核 1.2h;建議只跑 n∈[95,120] 分層 100k)。但死因對照:學髒 golden ✅ 繞開;realize 品質牆(2.35 vs bar 0.3)⚠ 未繞開;**prior 保真度負斜率 ❌ 直接抵觸**(品質已歸因 refine 段;LBR 同構反而支持不投 prior)。唯一翻案路徑=T2′(把 imitation 降維到 1 純量,同時繞三死因)。
**G-T3-1(零訓練,2 天,掛 T2′ 之後)**:修復後 golden 當「完美模型輸出」走 `refine_prediction` 通道,21 計分帶案 @0.3s。判準 noRT ≤1.10;**完美 prior 仍 >1.12 ⇒ imitation 整線判死**(瓶頸在 refine 搬運力,不在 prior)。

## 3. T1 oracle-prescreen 探針(最優先,半天)
儀器:`PARTNER_PSEL_DUMP=<path>`(default 空=dead branch),在 outs/ref_outs 匯合後 dump 每候選 (tid, channel, config_idx, hp, area, V, positions)(npz ~45MB)+ 選中標記。跑 3 reps @0.3s;離線用**官方口徑**重算每候選 cost(不可用內部 score,分母不同)。
判準:**G1** 池內 oracle 增益 <0.010 ⇒ learned-ranking 整類判死(含 racing/選擇器學習/配比重掃);**G2** M1/(M1+M2)<0.3 ⇒ racing 判死、全投 (d);**G3** 兩行修正(hp_ref 以實例統計估 golden 分母校正 + 死區 0.985→1.000)拿到 G1 的 ≥50% ⇒ 直接促轉兩行,不需任何 learned ranking。附帶產出:通道歸因升 A 級(memory 待辦 #2 交付)。

## 4. 排程
1) T1(0.5d)→ 2) G3(1d)→ 3) G-T2′-0(0.5d)→ 4) (d) 儀器+kernel(2d)→ 5) T2′ 訓練(2-3d,若 3 放行)→ 6) G-T3-1(2d,若 5 實得 ≥−0.010)。(c) 暫緩。
1-4 合計 EV 樂觀 −0.020~−0.030(1.169 → ~1.14-1.15)。**誠實上呈:單靠上述到不了 1.01——反事實表明三桶須同時近乎歸零(單桶完美最多 1.096)。**

## 5. 證據等級與待更正
A 級:§1 全部、harvest 表、frame 相關性、_wmedian 論證。B 級:斷崖 81%、P1/P2 數字、POOL46、kernel 倍率、golden 修復 0.08s。C 級:T2′ 實得 0.010-0.020、(d) 收益、G3 拿一半的猜測。
未驗證:通道歸因 B 級(T1 dump 坐實)、frame 因果性(oracle probe 是唯一判準)、`n_soft_den` vs `max_possible_violations` 一致性(tid 84 前科)。
**memory 更正兩則**:①軟約束全滿分 1.0722 → 0.3s 檔 1.0964 / 3.5 檔 1.0562;② learned C* 的「每 config 2-3× 退火」論證在 24-worker 架構下不成立。

關鍵位置:`src/solver/column_sa_legalizer.py`(選擇器 3731-3755、`_w_star_from_tags` 3570、pool init 3105)、`src/solver/layout_refiner.py`(`_wmedian` 576、`_median_shift` 583、`_axis_pass` 700、`run()` 3896)、`src/solver/refine_numeric_kernel.py`(Scope 1-80)。
