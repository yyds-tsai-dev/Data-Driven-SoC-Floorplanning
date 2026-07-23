# Budget 尾段掃描判讀(2026-07-23,Task 8)

Config:`PARTNER_BUDGET_MAX` ∈ {24(基線), 8, 6, 4};其餘 = 計分 env
(`VKILL_OFF=1 PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 PARTNER_OVERSAMPLE=4
PARTNER_TAG_ANCHOR_EXTRA=3`),checkpoint = `eval_retrieval_direct_control.pt`
(step 1,139,000)。全部 100/100 feasible。

| config | noRT_Q | alphaProj | beta~0.7 | beta~0.5 | sum_rt | tailQ(n≥100) |
|---|---|---|---|---|---|---|
| MAX=24(基線)| 1.1258 | 1.3383 | 1.4892 | 1.6474 | 487s | 1.1141 |
| MAX=8 | 1.1532 | 1.1275 | 1.2546 | 1.3879 | 332s | 1.1423 |
| **MAX=6(定案)** | **1.1596** | **1.0508** | **1.1692** | **1.2934** | **282s** | 1.1514 |
| MAX=4 | 1.2464 | 0.9952 | 1.1074 | 1.2250 | 215s | 1.2443 |

(alphaProj = deadline-bounded 假設下以 alpha field median 為參照的官方投影;
beta~X = field median ×X 的提速情境。)

## 判讀

1. **時間-品質彈性證實極低**:尾段 24→8s(3× cut)只掉 noRT_Q +0.0274;
   8→6 再 +0.0064。0712 的 budget×2 單點外推(每減半 ~+0.011)在 6s 以上成立。
2. **Direct cliff 現形於 MAX=4**:noRT_Q 突跳 +0.0868(vs MAX=6),tailQ
   1.1514→1.2443 — MAX=4 使 n≳84 全部被 cap,`PARTNER_DIRECT_MIN` 時間閘門
   開始關閉尾段最強的 direct 通道。超過計畫的 +0.03 cliff 門檻。
3. **定案:`PARTNER_BUDGET_MAX=6`**(cliff 條款)。alphaProj 1.3383→1.0508
   (−0.288),品質代價 +0.034。MAX=4 雖 alphaProj 最低(0.9952)且在 beta
   提速情境仍領先 ~0.06,但其品質回吐幾乎全部品質戰役進步,屬 cliff 邊緣的
   脆弱點 — 留給 DDIM25 + physics guidance + `PARTNER_DIRECT_MIN` 下調把
   direct 下限降低後,以「max4_ddim25」重掃驗證(預期屆時品質顯著優於 1.2464)。
4. 後續(T8 Step 3,等 guidance 鏈 T3-T6 落地後跑,避免 my_opt 編輯衝突):
   `budget6_pool46`(品質補償)與 `budget6_ddim25`(GPU 減步)兩個對照,
   之後 T9 guidance gate 都在 MAX=6 檔位執行。

證據:`artifacts/partner_eval/budget_scan_max{8,6,4}.json`
(基線 `cont_retrieval_direct_control.json`);判讀腳本
`scripts/probes/analyze_budget_scan.py`。
