# 2026-07-12 — Partner 管線時間中性優化:供給放大促轉(1.1263 → 1.1198/1.1213)

**目標**:只改 `partner/` 程式,將 no-runtime 分數自 1.1263([0711 驗證](../../artifacts/partner_eval/direct_v2_full100.json))壓向 ≤1.08。
**鐵律**(使用者 0712 指示):per-case 時間維持 partner 原設計(avg ~5s/max 24s),加時間的結果一律只當 headroom oracle、不可促轉。

## 促轉結果:T6「供給放大」

| 配置 | 全 100 案 no-runtime | avg runtime |
|---|---|---|
| 基線(bare defaults)| 1.1263 | 4.86s |
| **T6**(3 env vars)| **1.1198 / 1.1213(兩次複測,σ≈±0.0008)** | 4.87s |

```bash
PARTNER_PRESCREEN_V=1   # prescreen 排序加違規感知(貼牆距離+群組離散度)
PARTNER_NREF=12         # 直接預測精修槽 8→12(僅 n≥95 生效;column 重啟 16→12)
PARTNER_OVERSAMPLE=4    # GPU 採樣 2×→4×(與 pool SA 併行,48 樣本內牆鐘免費)
```

代表性單案:99 HPWL gap 0.046→0.005(權重最大案)、91 0.176→0.042、93 V 1→0(−0.058)。
機制:**候選供給品質 >> 局部搜索**——同模型同時間,讓更多且排序更準的預測拿到 full-budget 精修槽。

## 判死路線(死因入卷)

| 路線 | 全量效果 | 死因 |
|---|---|---|
| budget×2(oracle)| 1.1096 | 違規時間鐵律;證明殘餘違規大多可殺、SA 時間邊際仍高 |
| carve-25% 給 vkill(T1)| +0.0075 | SA 末段邊際 > vkill stage2 收益 |
| column-SA v_weight 退火(T2)| +0.0008 | 打錯引擎:尾段贏家 ~13/15 是 direct-refine 輸出(REFINER_DEBUG 歸屬實測)|
| refine worker v_weight 多樣性(T5)| wash | 偏置搜索不如換候選 |
| 供給放大過頭 NREF=16/OS=8(T7)| **+0.0706** | 128 樣本 GPU 批次 ~10s 卡在精修派工之前,worker 空等、精修全面崩潰 |
| T6 + 微 carve(1s)stage1 vkill(T8a)| +0.0001(違規 186→175)| 分數 wash;供給放大後殘餘違規對外科手術更抗性 |

## 新增 env 旋鈕(全部 opt-in,bare defaults = partner 原行為)

- `partner/my_opt_claude.py`:`PARTNER_BUDGET_SCALE/_TAU/_MIN/_MAX`(時間預算)、`PARTNER_OVERSAMPLE`、`PARTNER_PRESCREEN_V(W)`、`VKILL`(=1 啟用,搭配 `VKILL_CARVE/MIN_N/RESERVE_MAX/STAGE2_*`)
- `partner/legalizer_claude.py`:`PARTNER_NREF(_MIN_N)`、`PARTNER_REFINE_VW_MIX`、`PARTNER_VW_ANNEAL(_AT)`
- `partner/vkill_claude.py`(新檔):違規獵殺後處理(stage1 外科手術 + stage2 v_weight 深度重排),evaluator 容差全對齊,失敗靜默返回原結果

## 殘餘缺口與下一步

T6 佈局上分解:v→0 ≈ −0.05、hpwl→0 ≈ −0.03、area→0 ≈ −0.03;每案 SA/refine 抽籤方差 ±0.03–0.08 遠大於任何微調均值差 → **時間中性 solver 端天花板估 ~1.11–1.12**。
通往 1.08 的主路 = 模型端(別組 1.08 = 預測近乎免修):partner direct_v2 續訓,或以 fv2 快照作第二供給源(兩模型預測餵同一精修池,prescreen 濾弱票)。
次要撈分:90 型 column-案的自適應槽位(其 direct 預測固定帶 V=5-6,需 pre-sampling 的 instance-stats 訊號)、OVERSAMPLE/NREF 甜蜜點細掃。

## Perimeter 戰役(演算法級,判定:機制成立、整合未過、休眠)

診斷:最爛尾段案(89/79/90)共病 = **boundary 稠密(25-30% 塊帶牆碼)+ 牆載 85-95%**;89 的真病是**框架外溢**——preplaced 牆碼塊反推出意圖框架 [0,142]×[0,212],任何塊突出即整面牆碼雪崩(62 的 x1=141 與多數 142 不合,為題目自帶 V≥1 地板)。`refine_prediction` 的合法化階梯在塞不下時「放棄 tag pins」為殘餘 b 違規根因。

新機制 `_perimeter_pack`(`legalizer_claude.py`):牆線顯式裝箱(角落碼釘死、seed 序、soft 縮放階梯、帶狀障礙、locked 反推精確框架)——離線驗證 b→0-2 ✔。三種整合全敗:
1. column 釘死 ring:內部右溢一欄(`_layout` 總寬無硬上限,SA 困在溢出局部最優)
2. shelf 組合種子 → refine:min-displacement 保留爛全局結構,2.1-2.4
3. ring overlay(final/raw 預測皆試)→ T9 全量 1.1213(= T6 複測區間)wash,尾段 overlay 未贏擇優

跨解族 portfolio(我方 backbone 併入)oracle 僅 −0.0011,判死。未來落點:改 `refine_prediction` 的 `_seed_tags` 為聯合牆線裝箱(現為逐塊獨立 snap)。代碼閘控:`PARTNER_PERIMETER`(overlay)、`PARTNER_PERIMETER_COL`(column 釘死),預設全關。

證據:`artifacts/partner_eval/full100_{supply_amp,supply_amp_rep2,prescreenV,prescreen_vwmix,vwanneal25,timeneutral_vkill,budget2x_vkill,supply_amp2,t6_vkill_lite,t6_perimeter}.json`、`tail15_channel_diag.log`。
