# 2026-08-06 — Frontier-bracket promotion gate(0.3s 操作點促轉包終判)

## 背景

P 波三旗標(`PARTNER_EDGE_SEAT_V2` / `PARTNER_FRAME_WPIN` / `PARTNER_COL_NARROW`)與
`PARTNER_COORD_POLISH` 個別 gate 已過(見 2026-08-06 goal-101 研究定稿),但「polish
時間中性淨值」在兩條鏈上互相矛盾:

- combo 鏈(三臂 A/B/C ×2):C−B = **−0.0114**(2/2 勝)
- 定案鏈(A/P/Q ×3):P−A = **+0.0081**(1/3 勝)

## 矛盾根因:SCALE→runtime 校準漂移

兩鏈的 A 臂 config 完全相同(`PARTNER_BUDGET_SCALE=1.171e-4 / MAX=1.505`),realized
avg_runtime 卻在 10 分鐘內從 **0.290s 漂到 0.350s**。定案鏈的 A 因此比 P/Q 多吃
0.057s/案(恰好等於整個 polish 預算),P−A=+0.008 是時間不對齊的 artifact。
**結論:以「事前校準 SCALE」實作時間中性不可行**(deadline-SA 的固定 overhead 對機器
狀態敏感)。同時再度證實跨鏈絕對值不可比——判定必須同鏈交錯。

## 修正設計:frontier 括號(抗漂移)

同鏈 Latin-square 交錯 ×3 rep,三臂:

| 臂 | 旗標 | SCALE/MAX |
|---|---|---|
| A_lo | 全關 | 8.498e-5 / 1.22 |
| A_hi | 全關 | 1.171e-4 / 1.505 |
| C | seat + wpin + polish | 8.498e-5 / 1.22 |

A_lo/A_hi 畫出**同 rep 條件下**的 OFF 時間-品質前緣;C 的判定 = 其 realized time 上
前緣內插值與實際分的差。runtime 怎麼漂都不影響相對判定。共用 env:
dpmpp2 + flow8(euler, antithetic)+ SA/REFINE kernel numba + FAST_SETUP +
POOL_GATE=0 + tau12 + MIN=0.05 + NREF=6 + DM0.3。

## 結果(全 9 跑 100/100 feasible;加權 total_score_no_runtime)

| 臂 | noRT(3-rep mean±sd) | avg_rt | max_rt |
|---|---|---|---|
| A_lo | 1.1932 ± 0.0041 | 0.236s | 1.17–1.19s |
| A_hi | 1.1776 ± 0.0037 | 0.281s | 1.32–1.33s |
| C | **1.1681 ± 0.0023** | **0.292s** | 1.57–1.81s |

Per-rep 前緣校正淨值(負 = C 在 OFF 前緣下方 = 真增益):

| rep | slope(/s) | frontier@t_C | C | delta |
|---|---|---|---|---|
| 1 | −0.386 | 1.1685 | 1.1677 | −0.0008 |
| 2 | −0.425 | 1.1755 | 1.1660 | −0.0095 |
| 3 | −0.241 | 1.1768 | 1.1706 | −0.0062 |

- **校正淨值 mean −0.0055(sd 0.0044,3/3 勝)**
- raw C−A_hi(C 僅 +0.011s)= **−0.0095**(3/3 勝)
- 三鏈三角驗證:frontier −0.0055 / raw −0.0095 / combo C−B −0.0114 → 同向

## 判定

**促轉包 PASS**:seat + wpin + polish(SCALE rebalance 至 8.498e-5/1.22)在 0.3s
操作點是前緣下方的真增益,3/3 rep 同向,量測噪音為近期最低(臂內 sd ≤0.004)。
0.3s 操作點新定格:**noRT 1.168 @ avg 0.292s**(合規 ≤0.3s;max_rt 遠低於官方
免費尾段上限 2.4–3.6s)。

保留事項:
- `PARTNER_COL_NARROW`(P4)維持暫緩(分數 −0.0126 但機制歸因不成立)。
- B−A(seat+wpin 無 polish)的組合貢獻在兩條噪音鏈上讀 ~0,與個別旗標
  乾淨證據(各 −0.009,<0.001 噪音地板 session)矛盾未分離;不擋包級促轉,
  留作歸因實驗。
- polish 依賴 **scipy**(官方 requirements 無 scipy/numba)→ final repack 必加;
  numba 冷 JIT 首案 warm-up 護欄必做。

證據:session scratch `eval_fb_{alo,ahi,c}_r{1,2,3}.json`、`frontier_bracket_run.sh`;
前兩鏈 `eval_combo_*.json`、`eval_final_*.json`(session c70131c1 scratch)。
