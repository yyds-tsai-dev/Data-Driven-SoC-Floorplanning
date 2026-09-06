# 2026-07-29 DPM-Solver++(2M) direct sampler 促轉 + flow few-step Gate 0 判負

## 定案

**Promote：`PARTNER_DIRECT_SOLVER=dpmpp` + `PARTNER_DDIM_STEPS=10`（兩 rep 承重）；`PARTNER_DIRECT_MIN` 維持 2.0（1.5/1.0 掃描均 wash）。**
同期成對 A/B 對 DDIM25：noRT **−0.019~−0.021**、alphaProj **−0.027**，100/100 feasible 全程保持。

```bash
# 新定案 env（在 0723 定案 MAX=3.5+DM2.0 之上疊加）：
PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=10
```

## 實作

- `src/solver/direct_model_claude.py::sample_direct_dpmpp`：DPM-Solver++(2M)，data-prediction multistep 二階，midpoint 修正；一階步與 DDIM 代數等價，首步（無歷史）與最後積分步降一階（lower_order_final——少步時終端 h 大，二階外推會 overshoot）。保留 `sample_direct` 的逐步契約：anchor imposition → model call → aspect clamp → known 覆寫 → self-conditioning。無 guidance hook（guidance 已判死；`guide` 非 None 時呼叫端自動 fallback DDIM）。
- `src/solver/my_opt_claude.py`：`PARTNER_DIRECT_SOLVER` env 白名單分支（default `ddim` 不變）。
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
4. **SCFM 蒸餾線降優先級**：Gate 0 判負後蒸餾的角色從「免費減時」變「補 st2 缺口」，但 flow 段省時上限實測 <0.1s/case（st8 warm 0.117s），投報不足。蒸餾腳本已落地留存（flow worktree `src/solver/flow_distill_claude.py`，22 tests 綠，SCFM arXiv 2510.17858 方案，設計文件 `docs/superpowers/plans/2026-07-29-flow-distill-design.md`）——若 final 期 flow slots 擴編（NFE 總量變大）再啟用。
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

**`_fastsa_temp` 與 stall early-stop 均不促轉**（三輪同期 A/B 疊 dpmpp10 基準，9 runs 100/100 feasible）：fastsa dNoRT +0.0012/dProj +0.0018、stallstop dNoRT +0.0016/dProj +0.0015（dRT 僅 −2.4s/100 案）；tailQ 三輪符號一致退化（+0.001~0.002），判真實小退化非噪音。代碼留存 default off（`PARTNER_FASTSA_TEMP`/`PARTNER_SA_STALL_STOP`，src/solver/legalizer_claude.py +138 行；tests/test_partner_sa_cherry1.py 11 綠、206 tests 無回歸）。artifacts：`artifacts/partner_eval/cherry1_*.json`。

機制（結構性）：
1. **fastsa**：該溫度律為長鏈設計（src 給 3× 預算）；3.5s 檔位 worker chain span 僅 0.7-1.5s，stage-2 深淬火（1/6 預算 @T≈5e-5）把搜索鎖進 seed basin，stage-3 再加熱又拆掉成果。
2. **stallstop**：省下的時間無處可去——`_worker_refine` 照跑到 worker_deadline 且 `map_async` 等全體 straggler，case 牆鐘由最慢 worker 決定。**推論（任務 #9）：真·減時需 `_worker_refine` 收斂停止**，SA stall-stop 才能轉化為 RTF 收益。
3. cherry-pick 方向整體期望值下修（src/fork 生態不相容：長鏈+可膨脹天花板 vs 短鏈 deadline-bounded）；第二批（`_adaptive_reweight`/`_cfix_move`）降優先擱置。

## 補記 2（0729 午後）：refine stall early-stop 促轉

**`PARTNER_REFINE_STALL_STOP=1`（單獨）promote**：三輪同期成對（疊 dpmpp10 基準，`artifacts/partner_eval/refstall_*.json`，9 runs 100/100 feasible）：dNoRT −0.0014/−0.0033/−0.0101（均值 **−0.0049**）、dProj 均值 **−0.0040**、dTailQ 三輪符號一致負。家族最佳均值 on = 1.1259/0.8722。

機制（與 SA 版相反）：`_Refiner.run` 是兩類 pool worker 的終端時間井（`_worker_refine` 與 `_worker_solve` 的 stage-2 slice 都終結於它）；stall break 停掉已收斂的 phase 後，省下的時間在 refine_prediction 的**後續 pass/rung 內有去處**——所以牆鐘幾乎不變（dSumRt <1s）而品質受益。SA 版 stall 的省時無處去（同 worker 內無後續消費者），對照成立。

**聯動版（+`PARTNER_SA_STALL_STOP`）不促轉**：rep1 +0.0055 反彈、均值 −0.0017 弱於單獨版——SA stall 維持判負。

實作：`src/solver/refiner_claude.py`（+69，window=phase span×0.25 自縮放、eps=0.002 相對、default off bit 級不變）；`tests/test_partner_refine_stall.py` 綠、partner 全家 175 綠。

**0729 最終定案 env（本日三項疊加）**：
```bash
PARTNER_DIRECT_SOLVER=dpmpp PARTNER_DDIM_STEPS=10 PARTNER_REFINE_STALL_STOP=1
# 基座不變:PARTNER_BUDGET_MAX=3.5 PARTNER_DIRECT_MIN=2.0 + flow st8 + antithetic
```

## 補記 3（0729 傍晚）：flow v3 mib-hinge gate 判負，提交包定版

**v3 判負**：v3（mib-hinge，1M steps 訓完 16:04）成對同腳本背靠背 gate（乾淨環境，GPU 無訓練爭用）：v1 = 1.1356 vs v3 = **1.1577（Δ +0.0221）**，gate 門檻 ≤−0.003 完全不過（`artifacts/partner_eval/v3gate_v1_paired.json` / `v3gate_v3.json`，兩輪 100/100 feasible）。650k 中期 gate 的持平（0.8932）是下坡前兆；mib-hinge 配方蓋棺。**提交包 `submission/cadc1013.tar.gz`（FLOW_CKPT=v1）即最終定版**——不換檔、不重打。

**方差觀察**：今天 control 家族全日 range 1.1237-1.1356（≈0.012），遠大於先前標的 ±0.003 —— 3.5s 檔位的跨期方差含機器負載動態成分。紀律再確認：**促轉判定只採信同期成對 Δ**；跨期絕對值僅供趨勢參考。

（fast-worker 所跑 `v3gate_v1_control.json`（1.1381）因 env 組態不明棄用，以同腳本 `v3gate_v1_paired.json` 為準。）

## 補記 4（0730）：CSA-in-refiner（任務 #7）判負 — 座標殘差已近枯竭

CSF「換引擎」判死後的唯一倖存後代（`docs/design/2026-07-29-csf-analytical-prototype.md` §7）：**CSA 只做座標精修**，在 backbone 已合法化的拓撲上，繼承 V_rel/area_gap 不動，只攻 HPWL gap。天花板 tail `hpwl_gap→0` = −0.0219，過 −0.01 gate 需 **~46% 捕獲率**。

**三輪同期成對 full-100 A/B 判負**（疊 0729 定案 env：dpmpp10 + DDIM10 + `PARTNER_REFINE_STALL_STOP=1`；9 runs 全 **100/100 feasible**，`artifacts/partner_eval/csaref_*.json`，腳本 `scratchpad/csaref_ab_chain.sh` / 分析 `scratchpad/analyze_csaref.py`）：

| round | arm | dNoRT | dProj | dTailQ | dSumRt | d_tail_hpwl | d_tail_area | d_tail_Vrel | win/loss |
|---|---|---|---|---|---|---|---|---|---|
| rep1 | stall | +0.0054 | +0.0054 | +0.0060 | +0.8s | −0.0046 | −0.0008 | +0.0013 | 34/44 |
| rep2 | stall | +0.0007 | −0.0025 | −0.0010 | −0.7s | +0.0027 | +0.0003 | −0.0007 | 45/33 |
| rep3 | stall | −0.0004 | −0.0021 | −0.0005 | −0.6s | −0.0034 | −0.0048 | +0.0016 | 40/45 |
| rep1 | end | +0.0043 | +0.0032 | +0.0045 | +0.7s | +0.0098 | −0.0008 | −0.0016 | 43/36 |
| rep2 | end | −0.0009 | −0.0014 | −0.0015 | +0.6s | +0.0016 | +0.0018 | −0.0007 | 51/39 |
| rep3 | end | +0.0057 | +0.0038 | +0.0054 | +0.4s | −0.0061 | −0.0020 | +0.0022 | 40/43 |

均值 **stall +0.0019 / end +0.0030**（control 家族 1.1298-1.1316），gate `≤−0.003` 不過；tailQ 兩臂符號皆不一致（stall +0.0060/−0.0010/−0.0005）。牆鐘中性（±0.8s / 199s），符合減時鐵律但也代表沒有 runtime 側收益可換。

**死因（結構性，比「wash」更強的結論）**：
1. **捕獲率實測 ~3%，遠低於所需 46%**。tail `hpwl_gap` 成對 Δ = −0.0046 / +0.0027 / −0.0034（base 0.0558），均值 −0.0018，**符號不一致且與 control 自身跨輪散佈（0.0510-0.0613）同量級** —— 捕獲量測不出來，不是「小而真」而是「淹沒在諧波裡」。按天花板換算，3% 捕獲 ≈ −0.0007，比 gate 低一個量級。
2. **機制本身是真的，但在真實 layout 上餘量很小**。CSA 攻的是 `_axis_pass` 的座標下降盲點：它按拓撲序讓每組取自己的 1-D 加權中位最優，**從不為被拖動的下游鏈付 HPWL**，因此可以收斂到比起點更差的點且回不來。合成案（`tests/test_partner_csa_refine.py::test_csa_pass_escapes_the_coordinate_descent_fixed_point`）證實：median sweep 卡在 hp 49.0，CSA 把整條密排列走回 39.0。但 `_axis_pass` 的 backward `dmax` 最長路徑 pass **已經**讓零間隙鏈整體平移，真正只有聯合求解才拿得到的殘差（拖動定價的不對稱）在 column backbone 的飽和packing 上佔比極小。
3. 呼應 0707 定論：**tail 剩下的 hpwl_gap 0.056 是拓撲缺口（哪個塊放哪裡），不是座標缺口**；headroom 需 (order, shape) 聯合，單通道精修不可分解。CSA 是一個更好的座標 solver，但座標通道已近枯竭。
4. 兩種落點都測了，排除「時機」解釋：`stall`（迴圈內、median sweep 不動點處，與 squeeze 搶同一份 slack）與 `end`（終端、從 caller span 內 carve 出，squeeze 已先把 bbox 面積入袋）。end 臂 rep1 的 `d_tail_hpwl +0.0098` 顯示 carve 掉的 8% 搜索時間比 CSA 撿回的多。

**代碼留存 default off**（`PARTNER_CSA_REFINE=1` 啟用，`_WHERE` = stall|end|both，`_SHARE`/`_ITERS`/`_STEP`/`_DECAY`/`_MS`）：
- `src/solver/csa_coordinate_solver.py`（新，216 行）：`AxisHpwlObjective`（值 + 解析次梯度，向量化）、`ShiftPolytope`（refiner 自己的可行集 + retraction）、`csa_shifts`（Polak-Ribière、scale-free `c/‖p‖` 步長、幾何衰減取代論文 Q-table）。
- `src/solver/layout_refiner.py`：`_axis_constraints` 從 `_axis_pass` **逐字抽出**（兩邊共用同一多面體，任何分歧都是合法性漏洞）；`_csa_problem` / `_csa_pass`。
- **retraction 是唯一新的數學**，也是第一版失敗處：單用拓撲 clip 會把「鏈頭往右、鏈尾往左」的步整個壓成 0（實測直接把 CSA 釘死在座標下降不動點）；縮短步長也無效（飽和 packing 上接觸約束是緊的 ⇒ λ=0）。定案：Cimmino 平滑 → **兩側單向鏈修復**（down/up）→ 拓撲 clip 作合法性保證，由目標函數挑。
- 合法性：只動 xy 不動 wh（area 繼承）、移動限於 refiner 多面體（零重疊 by construction、已滿足 boundary tag 釘死、preplaced 凍結、cluster 剛性）、整個 pass 除非 `_key()` 嚴格改善且無重疊否則整體回滾、例外封裝。off 時 `csa_share == 0.0` 使所有謂詞成死枝，且 pass 不抽 `self.rng` ⇒ 決策邏輯與 rng 流 bit 級不變。
- 測試 `tests/test_partner_csa_refine.py` 19 綠（env 佈線、off 惰性、形狀/凍結塊/零重疊不變量、proxy 單調、目標 vs `opt._hpwl`、次梯度 vs 有限差分、投影可行性與冪等、逃逸不動點、終端落點不超時）；partner 全家 **243 綠**，全套 623 passed / 1 skipped。

**結案**：CSF 這條線（含本後代）全部關閉。剩餘 headroom 不在座標精修層。
