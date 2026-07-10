# Scale probe pt2 — ep10 完訓正式判定(2026-07-09/10)

**結論:存活(SURVIVES)但未入場(NO ENTRY)。** 容量縮放的收益在「縮放後可實現分」上持續兌付
(s32-realize 2.8473→**2.3499**,paired 91勝/9敗,尾部最強),但 ML 入場 bar
(**area_gap < +0.3**)在訓練跑滿 ep10 後仍差 2.3×(+0.687),且 0-2/100 案個別達標——
距離是**結構性**的(全體平移式改善,非尾巴噪音)。以入場為標準,本配方
(diffhgt_lite h256/l4 @ ns150000×ep10)到此收案;是否再投一級 scale(路線級決策)留 user。

## 受測 checkpoint

`checkpoints/diffusion_best_val_loss_0705_ns150000_ep10_diffhgt_lite_h256_l4_steps1000_acc16_bs4.pt`
(2026-07-09 12:56 完訓;best val loss 0.32499,ep10 收尾仍在刷新 best──val loss 僅為
training-health 訊號,判定全依 Evaluator Evidence)。判定前已釘副本防覆寫(0708 教訓):
`<sp10>/ep10_final_best_val.pt`。

## 儀器(與 ep5 batch4/5 逐位同規格)

`gen_decoder_probe.py --hints diffusion --diffusion-steps 32` + 環境
(`FloorSet/iccad2026contest` + `.env` + PYTHONPATH),100 案全量:

1. **s4-hpwl**(預註冊趨勢儀器):`--diffusion-samples 4`
2. **s16/s32-cost**(免訓練放大器):`--diffusion-samples {16,32} --select cost`
3. **兌現端 realize**(0708 落地、golden 無損驗證過):`--repair on --refine {on,selected}`
4. 全套 pytest

## 結果總表

| 儀器 | OLD h128/l2 | ep4 | ep5@22:40(快照最佳) | ep5 完訓 | **ep10 完訓** |
|---|---|---|---|---|---|
| s4-hpwl raw | 8.629 | 8.9496 | 8.6037 | 8.8439 | 8.8477 |
| s4 mean hpwl_gap | +1.309 | — | +1.292 | +1.265 | **+1.160** |
| s4 mean area_gap | +1.092 | — | +1.142 | +1.113 | **+1.034** |
| s16-cost raw | — | — | 7.4058 | — | 7.8677 |
| s32-cost raw | — | — | 6.9940 | 7.6687 | 7.5476 |
| s4-realize(repair+refine on) | — | — | 3.5794 | — | **3.0008** |
| s32-realize-selected | — | — | 2.8473 | — | **2.3499** |
| s32-realize gaps(hpwl/area) | — | — | +0.517/+0.829 | — | **+0.421/+0.687** |

參考錨點:golden-realize 天花板 **1.0472**、production(W1+E2 促轉後).env 基準 **1.2156**、
GNN 0514 v2 可實現 2.92(舊 ML 最佳)。

### 配對逐案(同儀器 ep10 vs ep5)

- **s4-realize:84勝/16敗**;band 平均 Δcost:n<60 −0.420 / 60-99 −0.604 / **n≥100 −0.682**。
  ep5 最爛案 case88 6.50→2.67。realize 轉換 84/100(repair 81 + evict 3),refine 接受 97/100。
- **s32-realize-sel:91勝/9敗**;n<60 −0.268 / 60-99 −0.395 / **n≥100 −0.508**。
  轉換 79/100(全 repair、零 evict/shelf),refine 接受 100/100(平均 Δ −0.035)。
- tail 不再有 10.0 飽和案(s32-sel 最差 3.39;raw 模式 tail 全 10.0──raw weighted total
  對尾部改善不敏感,是震盪的主因;mean gaps 是乾淨趨勢訊號)。

### 判定 vs 預註冊準則

1. **「跨越 OLD 8.629」(s4-hpwl)**:ep10 完訓 8.8477 未跨;多快照最佳仍為 ep5@22:40 的
   8.6037(單快照事件)。但未飽和訊號**首次雙優於 OLD 且全程單調**:mean hpwl_gap
   +1.160(OLD +1.309)、mean area_gap +1.034(OLD +1.092)→「清楚逼近」成立,**存活**。
2. **縮放後可實現分**:ep5→ep10(訓練量 ~2×)兌付 −0.50(2.8473→2.3499);best-of-N
   斜率仍陡(s4-realize 3.0008 → s32-sel 2.3499)。**新 ML 最佳 2.3499**,大幅優於
   GNN-0514-v2 的 2.92。
3. **入場 bar(area_gap<+0.3;fill>85% 未儀器化,area_gap 為約束項)**:**未達**。
   最佳組態 mean area_gap +0.687(bar 的 2.3×);逐案分布 min 0.269 / median 0.651 /
   p90 0.970,僅 2/100 案 <0.3。vs production:2.3499 / 1.2156 = **1.93×**。

### 外推(誠實版)

每 2× 訓練(ep5→ep10)兌付:realize total −0.50、area_gap −0.142(0.829→0.687)。
到 bar(+0.3)還需 −0.387 ≈ **2.7 個 doubling(若線性不衰減;經驗上會衰減)**;
即使到 bar,realized 還要 2.35→<1.22 才追平 production──以現斜率(遞減中)在賽期內
不可達。結論:再投一級 scale 的期望值為負,除非接受「長線資產、非本屆武器」定位。

## 兌現端工程(--repair/--refine)驗收:PASS

- golden hints 上 repair 0 觸發、1.0472(非破壞性確認,0708);ep10 上零 shelf floor、
  零 legal-fallback,100/100 feasible。
- 評分器閘門(refine 只在 cost 不升時接受)在兩顆 ckpt × 三組態下無一例外生效。
- pytest **381 passed, 1 skipped**(含工作區 E2-adaptive/Fast-SA 半成品與 probe 變更)。

## 儀器備註

- 本輪機器被外部 PPO 訓練佔 ~42/64 核(load ~130):**wall/ms-per-case 欄位跨輪不可比**
  (s32-sel wall 12402s vs ep5 1022s 全是競爭噪音),分數為確定性評分不受影響。
- diffusion 取樣無 seed 控制(0707 註記):跨 ckpt 的 fallback/轉換計數帶波動;
  paired cost 勝負(91/9)對此穩健。

## Artifacts

- 本輪(sp10 = `/tmp/claude-1100/-nashome-NVL4-vdalab-yyds-dev-Data-Driven-SoC-Floorplanning/3300b56b-f495-46b2-9ddc-aff67a97917c/scratchpad`):
  `ep10_{s4_hpwl,s16_cost,s32_cost,s4_realize,s32_realize_sel}.json`、對應 `.log`、
  `ep10_battery_status.log`、`ep10_battery.sh`、`ep10_final_best_val.pt`(釘)、`ep10_pytest.log`
- ep5 對照(sp5 = `.../e8f39261-51e0-4b60-a427-f9179edca334/scratchpad`):
  `ep5_s4_hpwl.json`、`ep5_s32_cost.json`、`ep5_s4_realize.json`、`ep5_s32_realize_sel.json`、
  `golden_realize_100.json`、`ep5_snapshot_2240.pt`(釘)

## Repro

```bash
cd FloorSet/iccad2026contest && set -a && source ../../.env && set +a
export PYTHONPATH="$PWD:$PWD/..:<repo>/src"
P="uv run python <repo>/scripts/probes/gen_decoder_probe.py"
CKPT=<repo>/checkpoints/diffusion_best_val_loss_0705_ns150000_ep10_diffhgt_lite_h256_l4_steps1000_acc16_bs4.pt
$P --hints diffusion --diffusion-checkpoint $CKPT --diffusion-steps 32 --diffusion-samples 4  --out s4.json
$P --hints diffusion --diffusion-checkpoint $CKPT --diffusion-steps 32 --diffusion-samples 32 --select cost --out s32.json
$P --hints diffusion --diffusion-checkpoint $CKPT --diffusion-steps 32 --diffusion-samples 4  --repair on --refine on --out s4_realize.json
$P --hints diffusion --diffusion-checkpoint $CKPT --diffusion-steps 32 --diffusion-samples 32 --select cost --repair on --refine selected --out s32_realize_sel.json
```
