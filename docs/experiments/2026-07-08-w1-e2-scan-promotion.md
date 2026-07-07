# 2026-07-08 W1 加寬 + E2 劑量/帶寬掃描 → 組合促轉(1.2297 → 1.2156)

延續 [2026-07-07-channel-matrix-e2-bands-bestofn](2026-07-07-channel-matrix-e2-bands-bestofn.md)。本輪:W1(portfolio 加寬)實作與量測、E2 第二確認與劑量/帶寬掃描、W1×E2 組合、促轉與驗證。

## 背景與旗標

- **W1 動機(官方 QA A2/A3,已錄 CONTEXT.md「Hidden-Test Hardware」)**:測試機 48 核 ICELAKE、允許 sample 內 multiprocessing;而 pool 公式 `min(12, cores//2)` + 12 條寫死 configs = 36 核閒置。加寬 = best-of-workers 順序統計,**零 wall-clock 代價**。
- 旗標(commit 4d79cb9,dormant):`FLOORSET_SA_WORKERS`(0=舊公式)、`FLOORSET_SA_CONFIGS`(0=舊 12 條;>12 以 C0±3 × v_weight × orientation 網格擴充);E2 旗標同前(`FLOORSET_TAIL_BUDGET_SCALE/_N`)。
- 本機 64 核 → W1 測試規模 32 workers / 32 configs。

## 量測(batch3,4-control 池化配對;controls 1.2279/1.2271/1.2316/1.2376,pooled 1.2311)

| 配置 | total | full 池化 Δ | n≥100 Δ(勝/21) | tail116 Δ | avg_rt | max_rt |
|---|---|---|---|---|---|---|
| E2-N100 **第二確認** | 1.2220 | −0.0091 | −0.0101(18) | −0.0048 | 7.2s | 45.2s |
| scale3/N100(劑量上探) | **1.2117** | −0.0194 | −0.0197(**20**) | −0.0042 | 10.4s | **69.2s ⚠** |
| scale2/N80(帶寬下探) | 1.2205 | −0.0106 | −0.0087(18) | −0.0028 | 8.2s | 47.0s |
| **W1 加寬(標準預算)** | 1.2246 | **−0.0064** | −0.0064(15) | −0.0034 | **4.6s(=control)** | 22.8s |
| **W1×E2 組合** | **1.2133** | **−0.0177** | −0.0160(**21/21**) | −0.0048 | 7.4s | 47.7s |

判讀:

1. **E2-N100 效應確認**(兩 run:−0.0128 / −0.0091,均值 ~−0.011)。
2. **劑量到 3× 仍單調**(−0.0194,20/21)→ SA 在 heavy band 遠未飽和;但 max_rt 69.2s 越過 60s 訊號線(evaluator 死參數 `timeout=60.0` 是 kit 中唯一「像時限」的訊號)→ **本地前沿參考、提交高風險組態**。
3. **帶寬下探 n≥80 無增量**(−0.0106 ≈ N100 的 −0.0091 + 噪音):n∈[80,100) 權重輕,不值得加時。
4. **W1 零代價收 −0.0064**(avg_rt 4.57 ≈ control 4.45、max 22.8s 不變)——official 分數零風險的純贏。文獻背書見 [研究報告](../research/2026-07-08-sa-time-quality-frontier.md):PARSAC 112 核同款獨立 restart;報酬遞減 ~N/2 → 32-40 workers 是甜蜜點。
5. **組合 21/21 全勝、−0.0177**,接近 scale3 增益但 runtime 只有 2/3。

## 促轉(2026-07-08,user 核准)

`.env` 新增:`FLOORSET_SA_WORKERS=32`、`FLOORSET_SA_CONFIGS=32`、`FLOORSET_TAIL_BUDGET_SCALE=2.0`、`FLOORSET_TAIL_BUDGET_N=100`。
**驗證 run(.env 預設)**:**1.2156**,100/100 feasible,avg 7.33s / max 46.6s——與組合量測 1.2133 同帶,相對歷史基準 1.2297 淨改善 **~−0.013**。

**RTF 風險備忘(提交前必須重審)**:官方 RuntimeFactor per-case 對全體提交中位數、慢懲罰不封頂(2× 慢 = 該案 +23%);E2 把最重 21 案的 runtime 加倍——若我方原本即在 field median,官方分數損失會遠超品質增益。W1 無此風險。**提交組態的 E2 開關留待提交期決策**;本促轉定義的是本地量測預設。

## 儀器備註

- control4(1.2376)明顯偏高,4-control 池化吸收;單 control 配對會高估 treatment 效應(N100 vs control3 曾讀 −0.0155,池化後 −0.0091~−0.0128)。
- 回歸測試(test_tail_budget + SA 等價 ×2)綠燈後才跑量測;所有 run 依序執行避免 CPU 競爭污染配對。

## 下一棒候選(依研究報告排序)

E2-adaptive 預算延伸(拿 scale3 增益、避開 69s worst-case)→ Fast-SA 三段排程 probe → `_stack_column` typed C-ext(先過 PyInstaller 打包 harness)。

## Repro

```bash
FLOORSET_SA_WORKERS=32 FLOORSET_SA_CONFIGS=32 bash scripts/eval_total.sh --output w1.json
FLOORSET_SA_WORKERS=32 FLOORSET_SA_CONFIGS=32 FLOORSET_TAIL_BUDGET_SCALE=2.0 FLOORSET_TAIL_BUDGET_N=100 \
    bash scripts/eval_total.sh --output combo.json
bash scripts/eval_total.sh --output promoted_default.json   # 促轉後 .env 即組合
```
工件:session scratchpad `e2_n100_confirm/e2_s3n100/e2_s2n80/w1_wide/w1e2_combo/e2_control4/promo_validation.json`。
