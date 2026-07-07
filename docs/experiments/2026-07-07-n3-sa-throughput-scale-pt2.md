# 2026-07-07 N3 SA 吞吐微優化(收案)+ scale probe 趨勢點 2(存活)

## N3:batched-move SA → profile 否決原設計 → 有界微優化落地 → 分數槓桿收案

**Profile 證據**(`profile_sa_throughput.py`,FAST_EVAL=1,GPU 訓練背景負載下):
熱點**不是** `_hpwl`/`_violations`(合計 11-20%),而是 `_layout_fast`/`_stack_column`/`_finish_layout`
(column 幾何重推 + 直譯器層開銷,65-85%)。batched-move 只能攤提 numpy 呼叫開銷、攤不掉每候選各自的
幾何重推 → **理論天花板 ~1.2×,原估 3-6× 不成立**。n=115 案 cache hit 僅 69%(5 個 locked obstacle
使 obstacle 欄位每 move 重算)。

**落地的微優化**([column_slicing.py](../../src/floorset_arch/legalizer/column_slicing.py),全部 bit-exact):
1. FAST_EVAL cache entry 擴至 9 欄:存 `col_top` + `top_item`(= `max(placed, key=yt)`)→ `_finish_layout`
   不再每 move 全 placed 掃描;legacy 4-tuple 分支保留,慢路徑 byte-identical。
2. obstacle 欄位二級 `(key, round(x,9))` cache(x 重現時直接命中)。
3. cache key genexpr → listcomp;locked-block pos 初始化向量化(`_locked_ids_np`)。

**驗證**:`test_sa_move_equivalence` + `test_sa_m3_m1c` 30 passed 1 skipped(FAST vs slow shadow 等價)。

**量測**(同工具重測):

| case | before | after | Δ |
|---|---|---|---|
| n=68(mid) | 3043 moves/s(hit 80.2%) | **3613(hit 81.9%)** | **+18.7%** |
| n=115(tail) | 1364 moves/s(hit 69.2%) | 1375(hit 67.5%) | **+0.8%(噪音內)** |

**判定**:exp(n/12) 權重集中於 n≥100,而該處增益 ~0(二級 x-cache 未命中:SA 的 x 值幾乎不精確重現;
瓶頸=obstacle 欄位重算 + `_stack_column` 分支密集內部)。換算 total < 0.001,低於量測地板(±0.006)。
**N3 作為分數槓桿 KILL**;深度重寫(numba JIT / `_stack_column` numpy 化)否決——PyInstaller 打包風險
(0705 註記:提交前必驗純 CPU 打包)+ 本日五殺確立的「調校即承重」定律。
微優化保留:bit-exact 零風險,mid-band 快 19% 對官方 RuntimeFactor(uncapped 慢懲罰)每 % 都有利。

## Scale probe 趨勢點 2:存活,斜率清楚逼近 OLD

同儀器(`gen_decoder_probe.py --hints diffusion --diffusion-steps 32 --diffusion-samples 4`),
checkpoint = `diffusion_latest_0705_ns150000_ep10_...h256_l4`(07-07 06:27,~ep3+;訓練進行中):

| | OLD h128/l2 練滿 | NEW pt1(ep2-3) | **NEW pt2(ep3+)** |
|---|---|---|---|
| weighted total | 8.629 | 9.997 | **8.950** |
| mean hpwl_gap | +1.309 | +2.618 | **+1.591** |
| mean area_gap | +1.092 | +1.906 | **+1.389** |
| legal fallback | 73/100 | 74/100 | **48/100(已大幅優於 OLD)** |

一個 epoch 內 hpwl_gap 對 OLD 的差距 +1.31 → **+0.28**;fallback 48 vs OLD 73 顯示 h256/l4 的座標
結構性更好(pin-order 衝突更少)。**依預註冊規則(ep5-6 前清楚逼近 → 存活)判:存活**。
正式判定點:ep5-6(~0709);屆時同儀器再測,若跨越 OLD → 路線 C 的座標品質路徑復活,
接 Track B v2 修復管線重測可實現分數(bar:area_gap<+0.3、fill>85%)。
趨勢 JSON:session scratchpad `trend_new_h256l4_pt2.json`(pt1/OLD 在 0706 session scratchpad)。

## 本日(0707)總帳

D1/A/B/N1/N2/N3 六個 Track-A 槓桿以量測判死或收案,Q2 規則 INVALID、Q4 已內建、Q5 無標的;
production 基準重確認 **1.2297**;疊層「調校即承重」定律三次獨立確立(D1 violation 稅、N1 hpwl 反噬、
N3 tail 吞吐地板)。**唯一存活的分數路線 = 路線 C(ML golden-like order),而 scale probe 趨勢點 2
是它今天收到的第一個正訊號。**
