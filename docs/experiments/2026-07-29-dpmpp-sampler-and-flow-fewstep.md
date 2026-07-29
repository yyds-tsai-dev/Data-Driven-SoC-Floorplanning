# 2026-07-29 DPM-Solver++(2M) direct sampler 促轉 + flow few-step Gate 0 判負

## 定案

**Promote：`PARTNER_DIRECT_SOLVER=dpmpp` + `PARTNER_DDIM_STEPS=10`（兩 rep 承重）；`PARTNER_DIRECT_MIN` 維持 2.0（1.5/1.0 掃描均 wash）。**
同期成對 A/B 對 DDIM25：noRT **−0.019~−0.021**、alphaProj **−0.027**，100/100 feasible 全程保持。

```bash
# 新定案 env（在 0723 定案 MAX=3.5+DM2.0 之上疊加）：
PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=10
```

## 實作

- `partner/direct_model_claude.py::sample_direct_dpmpp`：DPM-Solver++(2M)，data-prediction multistep 二階，midpoint 修正；一階步與 DDIM 代數等價，首步（無歷史）與最後積分步降一階（lower_order_final——少步時終端 h 大，二階外推會 overshoot）。保留 `sample_direct` 的逐步契約：anchor imposition → model call → aspect clamp → known 覆寫 → self-conditioning。無 guidance hook（guidance 已判死；`guide` 非 None 時呼叫端自動 fallback DDIM）。
- `partner/my_opt_claude.py`：`PARTNER_DIRECT_SOLVER` env 白名單分支（default `ddim` 不變）。
- `tests/test_partner_dpmpp.py` 7/7 綠：steps=1 與 DDIM bit 級一致、few-step 誤差不劣於同步數 DDIM（二階收斂）、anchor 精確、aspect clamp、NFE==steps、mask 歸零、同 seed 決定性。

## 全部 13 輪 full-100 數字（同期串行成對；環境含 flow v3 訓練佔 GPU 的已知爭用——因此**只採信同期 Δ**，不與歷史點 1.1385/0.8859 直接比絕對值）

| run | noRT | alphaProj | sum_rt | tailQ | band<60 | band 60-100 | band>=100 |
|---|---|---|---|---|---|---|---|
| control ddim25 rep1 | 1.1474 | 0.9014 | 203.2s | 1.1392 | 1.5654 | 1.2346 | 1.1449 |
| control ddim25 rep2 | 1.1503 | 0.9022 | 202.4s | 1.1429 | 1.5693 | 1.2399 | 1.1479 |
| **dpmpp10 rep1（促轉）** | **1.1289** | **0.8749** | 199.9s | 1.1191 | 1.5665 | 1.2289 | 1.1360 |
| **dpmpp10 rep2（促轉）** | **1.1294** | **0.8746** | 199.6s | 1.1182 | 1.5650 | 1.2366 | 1.1330 |
| dpmpp8 | 1.1314 | 0.8770 | 199.8s | 1.1218 | 1.5595 | 1.2330 | 1.1366 |
| dpmpp10+flow st2 | 1.1528 | 0.8921 | 199.3s | 1.1452 | 1.5685 | 1.2425 | 1.1532 |
| dpmpp10+flow st1 | 1.1700 | 0.9061 | 199.3s | 1.1611 | 1.5592 | 1.2563 | 1.1711 |
| dpmpp10+DM1.5 rep1 | 1.1276 | 0.8737 | 199.5s | 1.1178 | 1.5656 | 1.2287 | 1.1328 |
| dpmpp10+DM1.5 rep2 | 1.1291 | 0.8742 | 199.2s | 1.1191 | 1.5691 | 1.2300 | 1.1320 |
| dpmpp10+DM1.0 rep1 | 1.1259 | 0.8720 | 199.3s | 1.1164 | 1.5620 | 1.2297 | 1.1307 |
| dpmpp10+DM1.0 rep2 | 1.1293 | 0.8741 | 198.9s | 1.1184 | 1.5653 | 1.2330 | 1.1331 |

（rep 內方差：dpmpp10 σ≈0.0003、DM 變體 rep 差 0.002-0.003；control 本期漂高 ~0.009 vs 歷史 `direct12M_dfa.json`，係 v3 訓練 + 同期 CSF gate 負載爭用。artifacts：`artifacts/partner_eval/dpmpp_ab_control*.json`、`dpmpp10_ab*.json`、`dpmpp8_ab.json`、`dpmpp10_flowst*.json`、`dpmpp10_dm1*.json`）

## 機制發現

1. **dpmpp10 的收益 = DDIM50→25 同機制的延伸**：GPU 段 25→10 steps（~0.39s→~0.16s）→ 序列 GPU 段縮短 → refine workers 更早開工；二階 solver 在 10 steps 保持 25-step 解品質（一階=DDIM 等價性保證下界）。改善集中 60-100（−0.006）與 tail（−0.009~−0.012），小案持平——收益來自時間再分配，不是 raw 樣本品質。
2. **dpmpp8 開始退化**（+0.002 vs dpmpp10）：8 步以下二階截斷誤差進入可見區。10 = 甜點。
3. **flow few-step Gate 0 判負**：st2 +0.024、st1 +0.041，退化集中 tail（+0.026/+0.043）。機制：flow few-step 的 raw overlap 缺口（st1/st8 = 8.25×，st2 = 3.18×；位置 L1 僅 1.15×/1.05×——flow worktree 蒸餾設計文件 M1 測量）在 tail 案 refine 時間最緊處修不完。「自洽性>成分品質」定律的邊界：**成分品質差到一定程度（raw overlap 8×）就會穿透 refine 的修復能力**，st8 的成分品質確實承重。
4. **SCFM 蒸餾線降優先級**：Gate 0 判負後蒸餾的角色從「免費減時」變「補 st2 缺口」，但 flow 段省時上限實測 <0.1s/case（st8 warm 0.117s），投報不足。蒸餾腳本已落地留存（flow worktree `partner/flow_distill_claude.py`，22 tests 綠，SCFM arXiv 2510.17858 方案，設計文件 `docs/superpowers/plans/2026-07-29-flow-distill-design.md`）——若 final 期 flow slots 擴編（NFE 總量變大）再啟用。
5. **DM 軸全 wash**（2.0/1.5/1.0 家族 rep 均值全在 1.128-1.129 帶內）：dpmpp10 把 GPU 段縮到 ~0.16s 後，DIRECT_MIN 閘門對 budget 1-2s 案的邊際貢獻歸零——這些案的 flow+heuristic 種子已足夠，direct 通道開不開無感。單 rep 的 0.003 下沿（DM1.0 rep1）被 rep2 回歸修正，「調校即承重」再驗。

## 同日交叉判定（本輪 final 衝刺的其他戰線）

- **CSF analytical 換引擎路線 gate 判死**（`docs/design/2026-07-29-csf-analytical-prototype.md`）：n>100 20 案子集 CSF 9.93 vs column 1.12。結構性死因：合法化腿的 area_gap/V_rel 缺口（golden 座標+形狀餵入仍 area_gap 0.15+）；e^(2V) 指數槓桿 + exact-area aspect 彈性是 column backbone 的護城河。倖存賭注：CSA 作 refiner 內座標 solver（天花板 tail −0.022，任務 #7）。
- **legalizer fork 合流改判 cherry-pick**（`scratchpad/legalizer_fork_gap_0729.md`）：fork 與 src 平行演化非單向落後；全合流 10-16 人日高風險 → 逐項移植 `_fastsa_temp`/`_adaptive_reweight`/`_cfix_move`/stall early-stop，各自 A/B（任務 #6）。
- 論文調研總表：`docs/research/2026-07-29-beta-optimization-paper-survey.md`。

## 待辦

- [ ] flow v3 1M 訓完（~0729 14:00）→ 任務 #5 gate（dpmpp10 新基準 + v3 st8 + antithetic vs v1 同組合）
- [ ] dpmpp10 + antithetic 組合的乾淨環境複測（v3 訓完 GPU 空出後），刷新絕對值基準並供 beta/final 打包
- [ ] cherry-pick legalizer 組件（任務 #6）、CSA-in-refiner（任務 #7）

## 補記（0729 晚）：legalizer cherry-pick 第一批判負

**`_fastsa_temp` 與 stall early-stop 均不促轉**（三輪同期 A/B 疊 dpmpp10 基準，9 runs 100/100 feasible）：fastsa dNoRT +0.0012/dProj +0.0018、stallstop dNoRT +0.0016/dProj +0.0015（dRT 僅 −2.4s/100 案）；tailQ 三輪符號一致退化（+0.001~0.002），判真實小退化非噪音。代碼留存 default off（`PARTNER_FASTSA_TEMP`/`PARTNER_SA_STALL_STOP`，partner/legalizer_claude.py +138 行；tests/test_partner_sa_cherry1.py 11 綠、206 tests 無回歸）。artifacts：`artifacts/partner_eval/cherry1_*.json`。

機制（結構性）：
1. **fastsa**：該溫度律為長鏈設計（src 給 3× 預算）；3.5s 檔位 worker chain span 僅 0.7-1.5s，stage-2 深淬火（1/6 預算 @T≈5e-5）把搜索鎖進 seed basin，stage-3 再加熱又拆掉成果。
2. **stallstop**：省下的時間無處可去——`_worker_refine` 照跑到 worker_deadline 且 `map_async` 等全體 straggler，case 牆鐘由最慢 worker 決定。**推論（任務 #9）：真·減時需 `_worker_refine` 收斂停止**，SA stall-stop 才能轉化為 RTF 收益。
3. cherry-pick 方向整體期望值下修（src/fork 生態不相容：長鏈+可膨脹天花板 vs 短鏈 deadline-bounded）；第二批（`_adaptive_reweight`/`_cfix_move`）降優先擱置。

## 補記 2（0729 午後）：refine stall early-stop 促轉

**`PARTNER_REFINE_STALL_STOP=1`（單獨）promote**：三輪同期成對（疊 dpmpp10 基準，`artifacts/partner_eval/refstall_*.json`，9 runs 100/100 feasible）：dNoRT −0.0014/−0.0033/−0.0101（均值 **−0.0049**）、dProj 均值 **−0.0040**、dTailQ 三輪符號一致負。家族最佳均值 on = 1.1259/0.8722。

機制（與 SA 版相反）：`_Refiner.run` 是兩類 pool worker 的終端時間井（`_worker_refine` 與 `_worker_solve` 的 stage-2 slice 都終結於它）；stall break 停掉已收斂的 phase 後，省下的時間在 refine_prediction 的**後續 pass/rung 內有去處**——所以牆鐘幾乎不變（dSumRt <1s）而品質受益。SA 版 stall 的省時無處去（同 worker 內無後續消費者），對照成立。

**聯動版（+`PARTNER_SA_STALL_STOP`）不促轉**：rep1 +0.0055 反彈、均值 −0.0017 弱於單獨版——SA stall 維持判負。

實作：`partner/refiner_claude.py`（+69，window=phase span×0.25 自縮放、eps=0.002 相對、default off bit 級不變）；`tests/test_partner_refine_stall.py` 綠、partner 全家 175 綠。

**0729 最終定案 env（本日三項疊加）**：
```bash
PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=10 PARTNER_REFINE_STALL_STOP=1
# 基座不變:PARTNER_BUDGET_MAX=3.5 PARTNER_DIRECT_MIN=2.0 + flow st8 + antithetic
```
