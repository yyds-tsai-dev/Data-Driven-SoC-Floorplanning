# 2026-07-10/11 戰役日誌:七路排除 → ML 翻案 → fv2 + 1M 重訓啟動

**目標重定義(user):隊內目標 ≤1.08 = 別組真實達成的 total_score_no_runtime**(非 case
特調、~5s/案、不含 runtime)。現況 production 1.2156 → 需 −0.136,獎池 77% 在 n≥100。
本日以排除法掃完所有便宜/中價機制,全部判死;三條 user 情報把 ML 路線翻案;以
feature_version=2 + 吞吐修復重啟 1M 訓練。

## 一、判定總表(全部同儀器配對量測,勿重議)

| # | 路線 | 結果 | 關鍵證據 |
|---|---|---|---|
| 0 | scale pt2 ep10 正式判定 | 存活不入場(realize 2.3499,area_gap +0.687 vs bar +0.3) | 2026-07-10-scale-pt2-ep10-final-judgment.md |
| 1 | Fast-SA 三段排程 | ☠ +0.0005 噪音(附帶:s1.5 同品質 max_rt 46.7→33.3s,提交 RTF 保險) | sp5 scratchpad fastsa_*.json |
| 2 | E2-adaptive 預算 | ☠ +0.0068 反向,max_rt 68.9s 超線 | sp5 adap_*.json |
| 3 | ML per-case 擇優 | ☠ 只贏 3 小案,組合 −0.0001 | ep10_s32_realize_sel vs control |
| 4 | 方向 A:faithful/anchored 實現器 + (order,shape) 局部搜索 | ☠ **disproved**(往返稅 0.381→0.000 修復後,production=該類天花板;4案×3seeds 最佳僅 −0.0001) | 2026-07-10-direction-a-realizer-search.md |
| 5 | P2 QP 解析式 hint | ☠ placer 品質瓶頸(anchored 已消 58 shelf,仍 0/100) | 同上 |
| 6 | column 全開極限(時間無上限) | ☠ 飽和:scale4/N100/W48/C64 只得 **1.2129**(−0.0027) | colmax_s4.json |
| 7 | naive SP-SA(test99) | ☠ 2.909 vs 1.450;死因=preplaced+boundary 約束處理(golden 拓樸經 naive SP 亦 infeasible);HPWL 上行零證據 | scripts/probes/sp_sa_probe.py |
| 8 | 外框洩漏 → outline-guided backbone | ☠ DoA:`_choose_frame` 已 golden 級(面積 +0.11%/H 0.0%);area_gap=packing slack(小中案寬 11-14%) | scripts/probes/outline_leak_probe.py + frame 診斷 |
| 9 | FRAME_H_MULT 反失真 | ☠ 機制有效(n<60 area −0.021/hpwl −0.033)但權重陷阱:n<60 僅 0.6% 權重,oracle 上限 −0.0001 | frame_control/treat.json |
| 10 | pin 重建 hints | ☠ pins=I/O 邊界錨點非位置(78.7% 貼邊;位移中位 1.34·√area) | scripts/probes/pinrecon_hints.py |
| 11 | min-cut 分割驅動 | ☠ 結構性:golden-cut oracle 都輸 production 4/5(guillotine HPWL 劣) | scripts/probes/partition_probe.py |

**三定律入卷**:①「調校即承重」再證(D1 稅/N1 反噬 之後,anchored 自傷同型);
②「area_gap 大處=權重小處」(小案品質槓桿被 e^(n/12) 蓋死);③ 排除法完備後,
tail 缺口=**HPWL(0.19→需 ~0.10)+violation(0.034→需 ~0.01)**,area 已近 golden(+0.054)。

## 二、User 情報三條(改變戰局)

1. **1.08 = 別組真實分**,同尺(no-runtime)、非 case 特調、平均 ~5s/案。
2. **別組也用 diffusion,「model 幫助很大」**;官方 PDF 原文:*"Our internal
   experiments using a diffusion model … high-fidelity solutions in sub-minute
   intervals"*(FloorplanningContest_ICCAD_2026_v10.pdf)→ ML 機制雙重外部驗證,
   0709 判死屬外推悲觀,翻案。
3. **訓練吞吐異常=有東西每次重建**——命中(見四)。

## 三、diffusion 特徵稽核(user「寫太淺」= 半對)

真缺陷:①座標目標各向同性/恆正/frame-blind(vs N(0,1) 先驗失配);②全域條件
10 維有 4 維=常數 1.0(自除);③pin I/O 方向未編碼;④無 clean-sample 加權。
已寫對(勿重工):面積精確參數化 w·h≡a、boundary one-hot+loss、MIB/cluster/
preplaced 全接。`hgt_lite` 實為無 attention 的線性訊息+sum 聚合(深化=Tier 2)。

## 四、fv2 落地 + 吞吐修復(三關全綠)

- **feature_version=2**:T1 frame-aware 零均值座標歸一(estimator=面積閉合
  ws≈0.0289 + pins aspect;T1-lite fallback 在袋)、T2 全域條件修死項+補框架、
  T3 pin 方向特徵(frame-relative 質心偏移+四牆鄰近分數)、T4 dirty-order 加權
  0.35。版本護欄:legacy=v1 可載入、mismatch 明確報錯。
- **吞吐根因修復**:graph build(含 O(n²) pair 特徵)原在主進程單線程每樣本每
  epoch 重建;移入 DataLoader worker 路徑(_PreparedDiffusionDataset+collate+
  `.to(device)`)→ **13.5→76.7 graphs/s(5.7×)**;RAM 全量快取被否(1M pair
  張量 ≈220GB>128GB)。
- 關卡:diffusion 測試組 55 綠 → 全套 pytest **382 綠** → smoke 訓練無 NaN →
  probe 消費往返 5/5 feasible。

## 五、1M 訓練啟動(2026-07-11 凌晨)

Tag `0711_ns1000000_ep6_diffhgt_lite_h256_l4_steps1000_acc32_bs1_fv2`;
NOISE_SAMPLES=4、DIRTY_ORDER_WEIGHT=0.35、w24、cuda(L4);ETA ~24-42h。
監控:每 epoch val loss 回報+快照釘存(多快照取評測最佳)+錯誤警報。
**判定協議:promote 只認 `gen_decoder_probe --hints diffusion` realize 分數**
(對照:現役 ckpt 天花板 2.33、production 1.2156、目標 ≈1.08),val loss 僅健康訊號。

## 六、資產沉澱

- 實現器三模式:`--realize {compact,faithful,anchored}`(faithful=保真 tax 0;
  anchored=非破壞+可壓縮,portfolio-safe)。
- 探針七支:outline_leak / perturb / analytic_hints / pinrecon_hints / sp_sa /
  partition / anchored_sa_pilot(全部 CLI+JSON 證據,seed 固定)。
- 證據 JSON/log:session scratchpad `3300b56b-…/scratchpad/`(ep10 電池、colmax、
  各探針)+ `e8f39261-…/scratchpad/`(ep5 對照、fastsa/adaptive pairs)。
