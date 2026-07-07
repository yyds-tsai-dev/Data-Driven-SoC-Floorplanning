# 2026-07-07(晚)通道分解矩陣 + E2 帶寬掃描 + best-of-N — 五 probe 收案

延續 [gate0-order-channel-e2-tail-budget](2026-07-07-gate0-order-channel-e2-tail-budget.md)。五項:(a) Gate 0c 形狀通道、(b) column SA 形狀掛載點、(c) tail 天花板定價、(d) E2 帶寬/劑量、(e) best-of-N 推論縮放。

## (a) Gate 0c:形狀通道 — KILL,且完成「通道不可分解」矩陣

實作:`--shape-hints {self,golden}`(gen_decoder_probe.py,鏡像 --order-hints;`decode_shapes` 的 aspect 來源切換,`_legal_shelf_fallback` 維持 self 語義;sanity:golden/golden 逐位一致、self/省略等價)。

**decoder-realized 全量矩陣(weighted no-rt / legal-fallback)**:

| order \ shapes | production | diffusion(self) | golden |
|---|---|---|---|
| production | 5.34 / 33(D1) | — | **8.98 / 73** |
| diffusion(self) | — | 8.95 / 48 | **8.77 / 56** |
| golden | 9.21 / 79 | 9.37 / 74 | **1.0535 / 0** |

**核心結論(升級版 jointness)**:**自洽性 > 成分品質**。PP(5.34)勝過所有混合 cell——把 golden 成分(不論 order 或 shape)混進非 golden 幾何,比保持非 golden 自洽**更差**。(order, shape) 是不可分解的聯合通道;任何 piecewise oracle 注入全滅。0.18 headroom 只能由「同一個 coherent 的 golden-like layout」攜帶 → ML 必須整張圖學好(= scale probe),不存在單通道捷徑。

## (b) column SA 餵 golden aspect — 讀碼級 KILL(無掛載點)

`_unit_h(u,w)=eff_rigid_h+eff_soft/w`(column_slicing.py:872-875)、`_place_chunk_up` `bh=a/bw`:**column 內 soft block 形狀=面積/欄寬的衍生物**,主 SA moves(relocate/swap/reorder)不觸形狀,per-block aspect 僅存 refine 層微幅移動(已收案 +0.003)。餵 per-block aspect hint 需改造為列內異寬佈局 = 表示層重寫,非槓桿。此發現同時把 1.35×GT HPWL 天花板的機制講齊:**column 表示本身量化形狀**。

## (c) tail 天花板定價(golden-decode per-case rows vs 池化 control)

| 範圍 | weighted prize(若存在完美非 column 生成器) |
|---|---|
| 全量 | 0.1754(= 1.2289 池化 control − 1.0535) |
| n≥100(21 案) | **0.1358(77%)** |
| tail n≥116(5 案) | **0.0493** |
| test 99 單案 | **0.0258**(column 1.3530 vs ceiling 1.0303) |

獎池集中度確認;但「兌現」仍需 golden-like 聯合幾何——(a) 證明 oracle 拼裝不可行,realizable 路徑只剩 ML(scale probe)或「新非 column packer + 等強 constraint pass」(lever-A 級成本,violation 稅前車之鑑)。**E1 的便宜 pre-check 到此為止:獎池已定價,再往下就是貴的。**

## (d) E2 帶寬與劑量(三 control 池化配對;control 1.2279/1.2271/1.2316)

| 配置 | tail n≥116 | n≥100 帶 | 全量 | Σruntime |
|---|---|---|---|---|
| scale2/N116 ×2 組 | −0.0042/−0.0045(5/5×2) | −0.0013/−0.0037 | −0.0009/−0.0040 | +105s |
| **scale2/N100** | −0.0051(5/5) | **−0.0132(19/21,p≈2e-4)** | **−0.0128(total 1.2161)** | **+274s(445→719s)** |
| scale1.5/N116 | −0.0009(3/5)**低於 bar** | — | — | +50s |

**判定:`FLOORSET_TAIL_BUDGET_SCALE=2 + _N=100` 是本日最大槓桿(−0.0128,絕對值 1.2161 歷史最佳)**。機制:n=100–115 的 16 案原預算 9–20s 未觸 24s clamp,×2 為真加倍;劑量反應單調(1.5 不夠)→ SA 未飽和,scale>2 或 N<100 值得後續掃描。**促轉仍卡 RTF 提交策略**(慢懲罰不封頂、全落最重 21 案);旗標維持休眠。

## (e) best-of-N 推論縮放(現役 h256/l4 ckpt,ep3+;`--select cost` 新旗標)

| samples(cost-select) | weighted no-rt | fallback | wall |
|---|---|---|---|
| 4 | 8.86(hpwl-select 同日 8.95) | 60 | 109s |
| 16 | **7.84** | 40 | 260s |
| 32 | **7.33** | 38 | 451s(≈4.5s/案) |

**斜率陡且 32 樣本未見平緩**(16→32 仍 −0.52):diffusion prior 的樣本分布尾部遠好於均值,best-of-N + evaluator-faithful 選擇是免訓練放大器。**但對現役 ckpt 絕對值仍差 production 6×** → 不改變「等 ep5-6」的判定;價值在:**0709 判定時應同時跑 s16/s32+cost 掃描**,以「推論縮放後」的可實現分數評估,而非單樣本。儀器注意:diffusion 取樣無固定 seed,跨 run fallback 48/56/60 波動——criterion 效應(hpwl vs cost)無法乾淨歸因,未來要配對需加 seed 控制。

## 本輪總帳

- 通道分解:order-only ☠、shape-only ☠、oracle 拼裝全 ☠——**自洽聯合幾何是唯一載體**(理論收束完成,ML=唯一路,0709 判定)。
- 立即可用(等提交決策):**E2 N100 −0.0128**。
- 免訓練放大器(等 ckpt):best-of-N + cost-select。
- 工具沉淀:`--shape-hints`、`--select cost`、golden per-case rows(golden_decode_100.json)、池化 control 配對法。

## Repro

```bash
# (a) PG/DG cell
uv run python <repo>/scripts/probes/gen_decoder_probe.py --hints production --shape-hints golden --production-cache <sp>/prod_layouts_gate0.json --out ...
uv run python <repo>/scripts/probes/gen_decoder_probe.py --hints diffusion --shape-hints golden --diffusion-checkpoint <ckpt> --diffusion-steps 32 --diffusion-samples 4 --out ...
# (c) golden rows
uv run python <repo>/scripts/probes/gen_decoder_probe.py --hints golden --out <sp>/golden_decode_100.json
# (d)
FLOORSET_TAIL_BUDGET_SCALE=2.0 FLOORSET_TAIL_BUDGET_N=100 bash scripts/eval_total.sh --output n100.json
# (e)
uv run python <repo>/scripts/probes/gen_decoder_probe.py --hints diffusion ... --diffusion-samples 32 --select cost --out ...
```
