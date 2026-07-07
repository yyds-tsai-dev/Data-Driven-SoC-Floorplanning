# 2026-07-07 Lever B:SP window packer(M5 v3)— MEASURED KILL

## 假設與預註冊判準

Slicing packer 對近滿窗自由度不足(575 no_repack 窗);SP 窮舉覆蓋所有 overlap-free 拓樸
(含 pinwheel 等非 slicing packing),期望把部分 no_repack 轉成 accept,估 -0.003 ~ -0.008。
**Kill bar(預註冊)**:SP 帶來的 full-100 改善 < 0.003 → kill。

## 實作(tests 31/31 綠;程式碼保留、旗標預設 OFF)

- [src/floorset_arch/refine/sp_pack.py](../../src/floorset_arch/refine/sp_pack.py):k≤5 全 (k!)² SP 窮舉,
  沿用進場 shapes(不 resize → exact-area/MIB/dims 不變式建構上保證),numpy 向量化
  (precedence masks → Bellman-Ford longest path → ASAP/ALAP × 兩軸 4 種牆對齊 → anchored HPWL → top-N 去重),單窗 ms 級。
- [window_repack.py](../../src/floorset_arch/refine/window_repack.py):SP 候選走同一 `_consider` 路徑
  (guard/selection 不變,啟用只可能加候選);`FLOORSET_WINDOW_SP_PACK` 閘控 + `_SP_K`/`_SP_TOPN`。
- **單 run 自足歸因**:per-window JSONL `packer`("sp"/"classic")+ `classic_best_hpwl`(同選擇規則下
  classic 的 counterfactual)→ 不需兩次 run 相減吃 SA 噪音。
- 測試:[tests/test_sp_pack.py](../../tests/test_sp_pack.py)(語義/不變式/HPWL 排序/ALAP/決定性/閘控/歸因)。

## 結果(SP-on 全量,標準 eval_total.sh)

| 量 | 值 |
|---|---|
| Total Score (No Runtime) | **1.2357**(歷史 1.2298/1.2309,±0.006 噪音帶內) |
| window stage:tried / accepted / no_repack / soft_regress | 674 / 252 / 341 / 81 |
| **SP accepted / 全部 accepted** | **9 / 252** |
| accepted 窗總 HPWL 增益 | 221.7 |
| **SP 邊際 HPWL 增益(counterfactual 歸因)** | **0.6(佔 0.3%)≈ 0.00001 total-equiv** |
| SP 勝場組成 | sp_new(classic 修不動)4 窗、sp_better 5 窗 |
| k≤5 SP 適用窗 | 436 tried,SP 只贏 9 |

## 判定與死因

**KILL**(0.00001 ≪ 0.003 bar,差三個數量級)。死因:shapes 凍結下,近滿窗幾乎不存在
「能塞得下且 anchored HPWL 更低」的替代拓樸——window 價值的真正載體是 **aspect flex**
(classic slicing packer 已擁有),不是拓樸。這印證了檔頭原始觀察(BLF 無法 re-tile 近滿窗,
真自由度來自 re-slicing + flex),也與 D1 的教訓一致:column 產出的局部幾何已經是其拓樸下的
tight packing。**SP × shape-flex 組合擴展不再追**(SP 勝率 9/436 說明拓樸維度本身無礦)。

## 遺產

- 工具:單 run counterfactual 歸因欄位(`packer`/`classic_best_hpwl`/`n_sp_accepted`)留在 JSONL,
  之後任何 window packer 實驗直接複用。
- 旗標保持預設 OFF(未促轉 .env);程式碼與測試保留(31/31 綠,零風險)。
- 新鮮的 per-case 診斷(本 run):**n≥116 的 5 個 case 佔加權總分 37%**
  (test 99 獨佔 10.45%,hpwl_gap 0.3037 + v_rel 0.0597)——tail 集中度是下一波槓桿排序的關鍵輸入。

## Repro

```bash
export FLOORSET_WINDOW_SP_PACK=1
export FLOORSET_SLACK_REFINE_LOG=<path>.jsonl
bash scripts/eval_total.sh
python3 <scratchpad>/analyze_sp_windows.py <path>.jsonl
```
