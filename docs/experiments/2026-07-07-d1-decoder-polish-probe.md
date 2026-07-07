# 2026-07-07 D1 probe:decoder-as-polish 餵 production layout — MEASURED KILL

## 假設與預註冊判準(出處:[.claude/plans/2026-07-07-order-decoder-roadmap.md](../../.claude/plans/2026-07-07-order-decoder-roadmap.md))

- **D1**:把 production(column backbone + refine 疊層)的合法 layout 當 hint 餵 order-faithful exact decoder(`gen_decoder_probe.py`:order 抽取 → pin-aware longest-path compaction → exact-area → boundary snap → polish),期望對 layout 做無損 re-compaction + 牆違規修復,估 -0.01 ~ -0.04。
- **Kill 判準(預註冊)**:全量 weighted total 改善 < 0.002 → kill;`legal_fallback` > 5/100 → 先接 `repair_pin_order` 重測再判。
- **A(order-space local search)先決條件**:D1 ≥ -0.01,否則連帶 kill(inner realizer 無效)。

## 實作

- [scripts/probes/_probe_hints.py](../../scripts/probes/_probe_hints.py):新增 `ProductionHintProvider` — lazy import `solve_with_column_backbone`、`.env` 六旗標 parity(COLUMN_BACKBONE/SLACK_REFINE/ASPECT/VSNAP/FAST_EVAL/WINDOW_REPACK)、layouts JSON cache(retest 免重跑 SA)。
- [scripts/probes/gen_decoder_probe.py](../../scripts/probes/gen_decoder_probe.py):`--hints production` + `--production-cache`;**paired 報表**:同一份 layout 打 baseline 分數 vs decode 後分數 vs portfolio(min)(= strict-better gate 的部署語義,徹底消除跨 run SA 噪音)。
- 輸入建構與官方 harness 逐位一致:`iccad2026_evaluate.py` L855-888 傳**未切片**張量 + `opt_target_pos`(preplaced xywh / fixed wh);probe 的 `_opt_target_positions` 複製此建構。
- **golden 回歸測試**(3 cases):hpwl_gap −0.004、area_gap +0.000、0 fallback → driver 改動無破壞,decoder 對 golden 幾何依然無損。

## 結果(全量 100 case,paired,同 run)

| 量 | 值 |
|---|---|
| baseline(production layouts) | **1.2781** |
| decoded | **5.3367** |
| portfolio(min) — 部署語義 | **1.2781(deploy delta −0.0000)** |
| decoder 勝/敗 | **2 / 98** |
| legal-fallback | **33/100**(mean n=81,fallback 案 mean cost 9.68) |
| 耗時 | decode 21 ms/case;production solve 4.3 s/case;全程 448 s |

**非 fallback 67 案的分量分解(decoded − base)**:

| 分量 | delta | base → decoded |
|---|---|---|
| hpwl_gap | **+0.0025** | +0.2831 → +0.2856 |
| area_gap | +0.0189 | +0.1350 → +0.1539 |
| **v_rel** | **+0.0531** | **0.0506 → 0.1038(翻倍)** |
| cost | +0.1677 | 1.3551 → 1.5228 |

僅有的 2 勝均在低權重 case(idx=5 n=26:2.9026→2.6330;idx=35 n=56:1.3327→1.3167),weighted 貢獻 < 1e-3。decode 後 boundary_v mean 1.96、grouping_v mean 2.34;polish 在非 fallback 案 54/67 有 apply 仍不可挽回。

## 判定

- **D1 KILL**:deploy delta −0.0000,遠低於 +0.002 bar。
- **A 連帶 KILL**(預註冊先決失敗):order search 的每個 decode 候選都要付 soft-violation 稅(v_rel 翻倍 ≈ +0.15 cost)+ 33% hard-legal 崩潰率;起點即 +0.17(非 fallback 均值),seconds 級預算內不可恢復。
- **repair_pin_order 重測分支不啟用**:即使 33 個 fallback 全修好,非 fallback 子集平均仍 +0.168 cost——方向不可翻轉。

## 機制 post-mortem(修正研究階段的兩個推論)

1. ~~「compaction 解除 column 量化、擠掉欄間 dead space」~~ **不成立**:area_gap +0.019(production 在它自己的 order DAG 下已 ASAP-tight,無壓縮收益)。
2. ~~「slack-refined 位置被 ASAP 摧毀 → HPWL 大損」~~ **不成立**:hpwl delta 僅 +0.0025——order 抽取 + longest-path 重推座標對 HPWL **近乎無損,即使對 production 幾何**。
3. **真正死因:decoder 的 constraint pass(boundary snap + cluster 剛性)遠弱於 production refine 疊層(vsnap v1+v2 + violation-aware SA cost)** — 重推座標拋棄了 refine 層已調好的 boundary/grouping 結構,soft violation 稅無法由 snap 追回;另有 33% 案例 pin/ALAP/reshape 交互產生 hard-illegal → shelf fallback。
4. **golden→1.0535 依然成立**(本次回歸重確認):decoder 對 golden-like(ASAP-canonical、violation 結構單純)幾何無損。**0.18 headroom 的實現仍需 golden-like ORDER;順著 production order 重解碼不是路。**

## 側面觀察(open,不影響 paired 判定)

probe 同 run baseline **1.2781** vs 歷史 eval_total **1.2298**(+0.048,遠超 ±0.006 噪音帶)。輸入已證與 harness 逐位一致;候選解釋:背景執行的 CPU 資源限制/contention 使時間制 SA 同 wall-clock 少跑 moves。若未來要拿 cache 的 layouts 做「production parity」主張,需先做 1-command 檢證(同 8 case 前景 vs 背景重跑比 base costs)。

## 存活者與下一步

- **B(M5 v3:window repack 的 slicing packer → SP packer,k≤5 窮舉)**:按決策樹輪到它。D1 的死因**不**傳染給 B——窗內 repack 由 evaluator-faithful guard chain 把關(strict-better 才收),violation 稅會被 guard 過濾(最壞 no-accept 維持現狀);但含 boundary/cluster block 的窗預期 reject 率升高,期望值下修。
- **C / Track B ML**:本 probe 反而強化動機——realizer 對 golden-like order 有效(1.0535),ML 學到 golden-like order 即可兌現;等 scale probe 判定(ep5-6 規則)。
- 工具沉淀:`--hints production` + `--production-cache`(layouts cache 在 session scratchpad `prod_layouts.json`;全量 rows JSON `d1_production_full.json`)。

## Repro

```bash
cd FloorSet/iccad2026contest
set -a; source ../../.env; set +a
PYTHONPATH="$PWD:$PWD/..:<repo>/src" ~/.local/bin/uv run python \
    <repo>/scripts/probes/gen_decoder_probe.py --hints production --verbose \
    --production-cache <cache>.json --out <out>.json
# golden 回歸:--hints golden --cases 3
```
