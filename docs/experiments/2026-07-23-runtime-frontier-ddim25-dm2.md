# Runtime 前緣戰役:DDIM25 + DIRECT_MIN 解鎖 + MAX 下推(2026-07-23)

**結論:三個 env 旗標把 alpha 參照官方投影從 1.3383 壓到 0.9008(−0.44),no-runtime 品質僅讓 1.1258→1.1561,100/100 feasible 全程保持。新定案(待組合確認進 .env / beta 打包):**

```bash
PARTNER_BUDGET_MAX=3.5 PARTNER_DDIM_STEPS=25 PARTNER_DIRECT_MIN=2.0
# 其餘計分 env 不變:VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15
#                  PARTNER_OVERSAMPLE=4 PARTNER_TAG_ANCHOR_EXTRA=3
```

Checkpoint:`eval_retrieval_direct_control.pt`(direct_v2_cont step 1,139,000)。
投影方法:deadline-bounded(官方 runtime=budget),Cost_i 乘 `max(0.7,(rt_i/alpha_median_i)^0.3)`,λ∝e^(n/12);alpha median 見 docs/official/alpha_test/。

## 完整前緣表(全部 100/100 feasible)

| config | noRT_Q | alphaProj | beta~0.7 | beta~0.5 | sum_rt | tailQ(n≥100) |
|---|---|---|---|---|---|---|
| MAX=24(基線 = R4 control)| 1.1258 | 1.3383 | 1.4892 | 1.6474 | 487s | 1.1141 |
| MAX=8 | 1.1532 | 1.1275 | — | — | 332s | 1.1423 |
| MAX=6 | 1.1596 | 1.0508 | 1.1692 | 1.2934 | 282s | 1.1514 |
| MAX=6 + POOL46 | 1.1611 | 1.0588 | — | — | 285s | 1.1510 |
| MAX=6 + DDIM25 | 1.1314 | 1.0241 | 1.1396 | — | 281s | 1.1166 |
| MAX=5 + DDIM25 | 1.1385 | 0.9807 | 1.0913 | — | 252s | 1.1256 |
| MAX=4 + DDIM25(閘門關)| 1.2518 | 0.9992 | 1.1119 | — | 216s | 1.2512 |
| MAX=4 + DDIM25 + DM2 | 1.1363 | 0.9186 | 1.0221 | 1.1307 | 219s | 1.1243 |
| **MAX=3.5 + DDIM25 + DM2(定案)** | **1.1561** | **0.9008** | **1.0023** | **1.1088** | **200s** | 1.1482 |
| MAX=3 + DDIM25 + DM2(第二 cliff)| 1.3245 | 0.9994 | 1.1068 | 1.2243 | 188s | 1.3224 |

對照 alpha Top5(total):0.8789 / 0.9551 / 1.0197 / 1.0282 / 1.0997;我方 alpha 實測 1.9879。

## 機制發現(依時序)

1. **時間-品質彈性極低**(尾段 24→8→6s 只掉 +0.027/+0.006),證實 0712 budget×2 單點外推。
2. **第一道 cliff = `PARTNER_DIRECT_MIN` 硬閘門**(default 4.5,my_opt_claude.py:336):
   budget < 4.5 的案整個關閉 direct 通道 → MAX=4 品質跳 +0.087。`PARTNER_DIRECT_MIN=2.0`
   解鎖後 4s 檔品質完全恢復(1.2518→1.1363)。
3. **DDIM25(採樣 50→25 步)是純收益**:noRT_Q −0.028(1.1596→1.1314 @MAX6),改善集中尾段
   (tailQ −0.035)。機制:序列 GPU 段縮短 → refine workers 更早開工 + 閘門更易滿足。
   raw 預測品質差異被 refine 完全掩蓋。
4. **POOL=46 判死(wash +0.0015)**:本地 64 核已被訓練+eval 過訂閱;官方 48 核獨佔機
   或有不同,但無本地證據,按紀律不促轉。
5. **第二道 cliff @ MAX=3**:非閘門(3>2.0),是 refine/SA 的真實時間地板(3s 內一輪
   refine 做不完)。解鎖鑰匙 = SA/refine 內迴圈加速(numba,Task 12 進行中)。

## Physics guidance F4 判定:disproven(current design)

- 根因修復(HPWL 歸一化失衡,commit 0db74e4)後,raw overlap probe 全 7 案 −30%~−90%、
  viol 持平 — probe gate 全過。
- **但 full-100 劣化 +0.0070**(`budget4_dm2_pguide.json` 1.1433 vs control 1.1363)。
  Raw overlap 改善不轉化為 evaluator 品質:refine/SA 本能殺掉那些「好修的」overlap;
  guidance 花 GPU 時間修 refine 能修的東西並輕微擾動幾何(0707「自洽性>成分品質」再驗)。
- 依 evaluator-evidence 紀律記 **disproven at current design**;殘餘變體(不再燒 eval,
  留 beta 後):per-sample 歸一化修正 + HPWL 回歸、last-only(純末步 post-hoc)、
  boundary 稠密案限定觸發(reusable stats)。
- 代碼保留(`PARTNER_PHYSICS_GUIDE` 預設 off,bare defaults bit-for-bit 不變,
  12/12 tests + validate.sh 通過)。

## 證據

`artifacts/partner_eval/`:`budget_scan_max{8,6,4}.json`、`budget6_{pool46,ddim25}.json`、
`budget{5,4}_ddim25.json`、`budget4_ddim25_dm2.json`、`budget{35,3}_ddim25_dm2.json`、
`budget4_dm2_pguide.json`;probe:`artifacts/pguide/{quick,attrib,term}_probe.json`。
判讀:`scripts/probes/analyze_budget_scan.py`。
相關 commits:`3857982`(掃描)、`8c37eb9`(MAX6 判讀)、`961957a`..`85d0203`(guidance 鏈)、
`0db74e4`(HPWL 根因修復)、`8a821a3`(probe)。

## 待辦(接續)

- T12 numba(跑中):目標 3s 檔位解鎖(投影 ~0.86-0.88)+ 3.5s 檔品質直升。
- flow v1 訓完(~07-25)→ 修過 S-1 的 F3 probe → D/F/DF gate(整合已就緒,commit b44713a)。
- T13 beta 打包:op_wrapper.py 以本頁定案 env 為 setdefault;含 seed 方差複測
  (歷史 σ≈±0.0008,本頁差距均 >> noise,但打包前對定案組合複測一次)。
- beta 風險註記:beta 為 hidden cases + 獨佔機器,alpha median 僅為 field 曲線參考;
  MAX=3.5 在 beta~0.5(field 提速一倍)情境仍 1.1088,穩健。
