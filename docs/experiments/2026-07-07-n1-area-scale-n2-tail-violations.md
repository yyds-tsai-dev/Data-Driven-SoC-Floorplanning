# 2026-07-07 N1 面積縮放 probe + N2 tail 違規法醫 — 雙 MEASURED KILL

Post-B 排序(roadmap §8)的前兩名候選,同晚量測,雙殺。production 基準經 control run 重確認 **1.2297**。

## N1:FLOORSET_AREA_SCALE(±1% 面積容忍利用)— KILL

**假設**:官方容忍雙邊(PDF Eq.1 `|wh−a|/a ≤ 0.01`;evaluator L253-254 對稱絕對值),soft block 面積 ×0.991 → die bbox 與絕對 HPWL 同縮 → 估 −0.004~−0.008。

**實作**([column_slicing.py](../../src/floorset_arch/legalizer/column_slicing.py) `_resolve_shapes()` 後):`kind==0` 縮放;排除 fixed/preplaced(kind 1/2)與含 rigid 成員之 MIB group 的全部 soft 成員;全 soft MIB group 均勻縮放保 shape 一致。旗標預設 1.0(零行為變化),回歸 54/54 綠。

**量測**(scale=0.991 vs control=1.0,同日同機,100 案 per-case 配對):

| 量 | 全 100 案 mean(勝場) | n≥100 mean(勝場) |
|---|---|---|
| 縮塊生效 | block 面積比 golden median 0.9926 ✓ | — |
| area_gap delta | **−0.0139(75/100)✓** | −0.0101(17/21)✓ |
| **hpwl_gap delta** | **+0.0080(43/100)** | **+0.0212(5/21,sign test p≈0.013)** |
| v_rel delta | +0.0001(中性) | — |
| **加權 no-rt total** | **1.2350 vs 1.2297 = +0.0052(反向)** | |

**死因**:面積機制照設計運作,但 HPWL 在權重主體(n≥100)**系統性**劣化 +0.021,α=0.5 折算(+0.0106)> 面積贏(−0.005)。
**機制解讀**:production SA+refine 疊層是在 exact-area 幾何上調校的 local optimum(column caps、aspect ladder、snap 容差隱含 100% 面積假設);全域 1% 擾動把大 case 的搜索 de-tune。p2b terminal 拉遠效應(佔比 median 11%)僅能解釋 +0.001~0.002,非主因。**與 D1 同課:疊層調校是承重結構,天真的全域擾動付稅。**
**不追 rescue 變體**(frame 重定位/子集 gating):加權總分只由 tail 決定,而 tail 正是劣化處;期望值低、每次驗證一輪全量。

旗標休眠落地(同 M1c/M3/SP_PACK 慣例)。

## N2:tail 違規法醫 — KILL(0705 結構性判定獲 case 級 golden 對照確認)

**假設**:Eq.2 的 e^{2v} 乘法 × exp(n/12) 權重 → test 99/89/88/92 四案違規全清值 ~−0.036。

**診斷**(SP-on run 的 comparison PNG,零重跑):

| Case | 我方 soft 違規 | **golden 自身** |
|---|---|---|
| 99 | 5 | **4** |
| 89 | 6 | **7**(我方更好) |
| 88 | 7 | **7**(平) |

**結論**:tail 殘餘違規是 over-constrained instance 的結構性違規——golden 同樣帶著。可收頭寸僅 case 99 的 ~1 個(×權重 ~−0.003,可修性未知)。理論獎池是海市蜃樓。診斷成本:3 張 PNG。
**附帶確認**:三案肉眼皆見 whitespace 洞 + 細條 sliver 疊堆 + cluster 拉成整條 column(column 量化簽名)——tail 的稅在 HPWL 拓樸/面積,唯路線 C 可攻。

## 收斂後的 Track-A 現況(0707 夜)

D1/A/B/N1/N2 五連殺 + Q2 規則 INVALID + Q4 已內建 + Q5 無標的。**Track A 僅存 N3(batched-move SA 向量化,−0.005~−0.008,3-5 天中風險)**;結構性 headroom(1.2297 → ~1.05)唯路線 C(pair-cost head + blackbox-solver differentiation),等 scale-probe 判定。

## Repro

```bash
FLOORSET_AREA_SCALE=0.991 bash scripts/eval_total.sh --output scale.json   # N1
bash scripts/eval_total.sh --output control.json                            # 配對 control
# N2:artifacts/eval_v11/floorplans/total_20260707_082831_2834378/top_cost_rank_*.png
```
