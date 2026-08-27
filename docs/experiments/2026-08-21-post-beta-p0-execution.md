# 2026-08-21 Post-beta P0 執行記錄(新機首夜)

**目標**:roadmap(`docs/research/2026-08-21-beta-analysis-and-final-sprint-roadmap.md`)的 P0-B/C/D + P1 開工;P0-A(final 打包)依指示延後。
**環境**:遷移後新機(48 核、4×H100 共機,load 38–57;NAS home)。本夜全部為 **column-only**(`DIRECT_OFF=1 PARTNER_FLOW_SLOTS=0`)——模型 checkpoints 仍在舊機(LFS 未轉移)。所有分數 = 官方加權 no-runtime,本機口徑,絕對值不可與舊機比;判定一律同鏈交錯 + per-case paired bootstrap。

## 1. 環境 bootstrap(完成)

- FloorSet submodule 自 GitHub 還原(pinned SHA 吻合);`LiteTensorDataTest`(官方 100 validation)在 submodule 內,eval 全通。
- `FloorSet/floorset_lite`(1M 訓練集)自 HuggingFace 重新下載(6.6GB tar,100 workers)——**不必等舊機**。
- git-lfs binary 裝至 `~/.local/bin`(repo LFS 物件仍缺,見 §8)。
- 量測方法:單跑 σ=0.0125(5 reps,遠差於舊機 0.004,load 汙染);同鏈交錯後 arm 內 σ=0.001–0.004。分析工具:job scratch `analyze_pairs.py`(weighted paired delta + case bootstrap CI)。

## 2. P0-C 預算重校(本夜主戰果)

Beta floor 發現(rt_adj ≥ 0.7,rf ≤ 0.3046 免費)驅動的預算實驗序列,全部同鏈交錯 3 對:

| 實驗 | 曲線 | paired Δ vs 前一檔 | 95% CI | 判定 |
|---|---|---|---|---|
| b1(κ=0.20 常數曲線,MAX 未護尾) | SCALE 0.0718/TAU 51.5 | −0.0017 | [−0.0155,+0.0077] | wash——**中段大勝被尾段砍預算抵消**(教訓:TAU 51.5 在 n≥110 低於舊 clamp 1.22) |
| **b1v2**(尾段護住) | SCALE 0.119/TAU 51.5/MAX 2.0 | **−0.0203** | **[−0.0310,−0.0111]** | 3/3 勝;HPWL −0.0192 主導;mega 案 99/88/89 全改善 |
| **b1v3** | SCALE 0.14 | **−0.0067**(vs b1v2) | [−0.0133,−0.0008] | 3/3 勝,HPWL 驅動 |

關鍵單案證據:中段饑餓懸崖——case 61(n≈82)從 0.05s 解鎖到 ~0.3s 單案 **−0.216**。

**但 hidden 期望值否決單一曲線**:以 beta median 表(QA A12:final 與 beta 同一組 hidden cases;id↔n 一一對應)+ 機器係數 M(本機→contest ≈2.1,由 beta 52.07s/舊機 29.3s 與本機/舊機速度比推得)+ median 漂移 D 的 3×3 情境格點評估 `E[cost_noRT × rt_adj]`:

- b1v3 在中性情境**反輸** b1v2 +0.018(mid-tail 超 floor 25–50%,rt_adj 罰 +5–13%);
- 悲觀角(M=2.4, D=0.8)連 b1v2 都輸 base;
- **per-n 逐檔選優(3 個實測檔位)在全部 9 情境贏 −0.01~−0.02**。

**定案:`PARTNER_BUDGET_TABLE`**(新 env,100 個 per-n 秒數,n=21..120;`contest_optimizer._time_budget` 讀表,未設/壞值 fallback 原曲線,byte-identical)。表值 = 情境加權期望選檔(M∈{1.8,2.1,2.4}×D∈{0.8,1.0,1.3},中心加權),寫於 `artifacts/p0_newbox/budget_table_ev.txt`。離線期望:中性 −0.013,最壞角 ±0.000。確認鏈(base/table/b1v2 ×3)見 §9 補記。
過度平滑版(rolling-median + 0.9×floor cap)實測期望反而劣化(−0.003/+0.007)——建模工件疊加,棄。

## 3. P0-B violation 工作

Deep-reasoner 審計(以 0806/0807 工件 + 本夜新 audit)改寫作戰圖:

- **MIB=0**(official100,全帶)——不花工。
- Boundary ≈70%/grouping ≈30%;74% 加權違規在 n≥100。
- **P2(FRAME_WPIN)早已於 08-06 出貨**;殘量真因 = W* 開環目標(util 猜 0.96,實際 0.88–0.97)+ 單邊 widen retry。
- **結構性發現:tid 88/89/99(≈38% 加權違規)雙軸被 preplaced tag 釘死,隱含 frame 需 96.1–98.1% util,我們 91.8–93.3%——非 seating 問題,是死空間問題**;唯一槓桿 = COL_NARROW/P4 類。

實測(全部同鏈交錯 3 對):

| 變更 | 在 b1v2 上 | 在 b1v3 上 | 判定 |
|---|---|---|---|
| `PARTNER_COL_NARROW=1`(重驗) | −0.0033 [−0.0094,+0.0020];case 89 單案 −0.061(機制吻合) | +0.0004 [−0.0031,+0.0034];v_rel −0.0014 但 HPWL/area 付回 | **不促轉**;機制真實但高預算下吞吐稅抵消 |
| `PARTNER_CLUSTER_GLUE=1`(新碼,anchored-cluster 側貼+最近縫隙) | — | +0.0037 [−0.0022,+0.0101];v_rel −0.0011 但 HPWL +0.0071 | **不促轉**;維持 default off。v2 方向:僅在 champion 有 grouping violation 時選擇性重試 |

落地的正確性/儀器碼(旗標 off,off-path byte-identical,測試綠):

- `PARTNER_PICK_EXACT_V`:`_pick_best` 改用 `_violations_exact` 排序(修 1e-7 容忍性計數器)。**本機 arms-off 時 no-op,等 ckpt 回來再量測。**
- `PARTNER_VAUDIT_JSONL`:per-case 六階段違規帳本 + 型別明細(atexit flush)。
- `tests/test_partner_pick_exact_v.py`(4)、`tests/test_partner_cluster_glue.py`(5)、guard 套件 134/164 綠。

## 4. P0-D 落差診斷(本夜最大認知更新)

`scripts/probes/pseudo_hidden_eval.py`(新):自 floorset_lite heldout 抽樣(重用 `BandFileSampler._instance`/`split_for_id`/官方 scorer),輸出官方口徑 JSON + 違規型別分項。

1. **v1 校準(1-per-n,100 案)**:pseudo 1.4494 vs official100 同 env 1.290 → +0.16,酷似 beta 落差 +0.156。分解:quality factor ×0.995(持平),**violation factor ×1.126 —— 全部來自違規,其中 v_mib=3.68/case(official100 = 0)**。
2. 追查:訓練集 MIB group 成員面積 spread 0.79–0.91(差 9×)→「等維 + 硬 1% 面積容差」數學上不可滿足;validation 的 MIB group 面積同質(spread 0)。**rank 1 hidden raw 1.0027(距下限 0.27%)⇒ hidden 不可能有結構性不可避免違規 ⇒ 訓練集 MIB 異質是 data 工件,不轉移。**(面積聚類可回收值也僅 3.68→3.57,無利可圖。)
3. Harness 加 `--mib-max-spread 0.02` 過濾(default on)。**過濾後 pseudo gap 只剩 +0.022(×1.017),且 violations 反而更好(0.0265 vs 0.0405)** —— solver 對新 instance 泛化乾淨。
4. **Beta +0.156 落差假說重排**:①contest 機吞吐 × mid-band 饑餓懸崖(beta 包 mid-band 只給 0.05–0.15s;懸崖單案可 −0.2;b1v2/表已正解)②hidden 尾案結構(88/89/99 型雙軸釘死案佔比可能更高)③median 表口徑的 rt_adj 細節。原「機器慢只值 0.011」的推論因忽略懸崖非線性而**低估**。
5. 過濾副作用:訓練集含 MIB 的 instance 幾乎全異質 → 過濾後樣本以無 MIB instance 為主(1-per-n 只湊到 50/100 n 值)。可接受:hidden 的 MIB 對我們本來就零違規。

## 5. ePlace arm(user 指定方向)— 完整證據鏈,終判 KILL

- Paper 調研(12+ 篇/5 repos):CSF(n100 GP 0.22s/50-200 iters,C++)、PeF(soft 寬度一級變數)、Cortadella 2026(convex log-legalization;AR [1/3,3] 同題)、DREAMPlace 3.0 region 機制、MAGICAL 對稱組(=MIB 共享維度模板)。舊 kill(CSF 需 2e4 iters)實為其合法化階段;GP 只要 50–200 iters——預算重審後可行。
- v0(`partner/eplace_arm.py` + probe):負訊號(median HPWL ratio 1.72、3/10 發散、overlap 壓到 0.001 = 解錯 tradeoff)。
- **deep-reasoner 數值修復**:根因 = ①項間無正規化(overlap 力/HPWL 力 = 1.3e3–1.1e4,Adam whitening 下 HPWL 貢獻 <0.1%)②soft-shape 梯度漏 `dh/dw` 項 → 全部寬度單調塌到 1:3 clip ③隨機 init(改 QP wirelength seed)④sign-descent 無幅度。修復後 raw probe **全過**:median ratio **0.7458**(21/21 ≤0.95)、overlap 9–18%、bbox 反而小 4%、0 發散、4 configs 0.285s/case。
- **決勝實驗(合法化後)**:餵入產線同款 `refine_prediction` 消費端(0.5s/候選,ids 79–99)→ **21/21 全敗,weighted delta +0.31**;精修後 HPWL ratio 只剩 0.9–1.05,官方 cost 被 violation/area 反噬(refine 路徑無法重建 boundary/grouping 結構——與 C-2「diagonal separation = refiner 簽名」一致)。
- **終判:KILL(此整合路徑)**。機制 = 賽季核心教訓重演:「raw 幾何更好,只有後端能保留其拓撲差異才有價值」。rank-3 的 ePlace 成立是因其整個後端圍繞 analytic 輸出構建;替代路(為 ePlace 建專屬合法化器/convex log-legalization 後端)= 週級工程,決賽窗內不可行。scaffold 保留(kernel+兩個 probe+JSON)。
- 附帶全 repo 適用發現:本機 `np.linalg.solve` 在 116×116 系統要 576ms(OpenBLAS 執行緒病理;`OMP_NUM_THREADS=1` 0.077ms、`cholesky` 0.57ms)——repo 內小型 dense solve 一律用 cholesky 或關執行緒。

## 5b. VAUDIT 六階段帳本(100 案,b1v2 config)

| stage | weighted V |
|---|---|
| 0 column champion(pre-seat) | 2.177 |
| 1 edge_seat 後 | 2.064 |
| 2–4 pick/polish/final_seat 後 | 2.064(**零回收**) |
| 5 tag_compress 後 | 2.061 |

**Violation 質量在建構期定型;四個外科 post-pass 合計只回收 5%** —— 修復空間只在建構期(B.3 閉環 frame targeting、死空間)。Final 型別:bnd 1.60(codes:top 39/bottom 28/corner 45/right 19/left 7)、grp 0.46、mib 0。無任何 stage 使 V 上升(guards 有效)。

## 6. TFDL(5.6-sol 續作)狀態

Task 5 精確配方已抽出(exact CLI/gates/hash 綁定)。**硬阻塞於舊機資產**:訓練源 ckpt `partner/checkpoints/direct_v2_cont/eval_step1p2M.pt`(sha256 綁定,不可重訓替代)、`submission/cadc1013/checkpoints/*`(在 LFS tar 內)、G0 語料 `artifacts/icdc_topology/` + `artifacts/icdc_g0_v2_area_full/`。G1 同阻塞。

## 7. 其他

- 兩份 evaluator(scripts/ vs submodule)不同 md5 = 17 個 topology 測試失敗根因。**單檔跑 831/831 綠;全套跑重現同樣 17 個失敗**(跨測試 import 順序:先前測試把 submodule 路徑放進 sys.path 後,frozen contract 解析到錯誤副本)——與 handoff 基線(2,260+17)完全一致。**今晚全部改動 0 新增失敗**(2026-08-22 全套:2,261 passed / 13 skipped / 17 failed,+9 新測試全綠)。根修 = scorer loader 改 path-explicit 載入 `scripts/iccad2026_evaluate.py`(backlog)。
- NARROW/GLUE/exact-V/VAUDIT/BUDGET_TABLE 全部 default-off/unset,工作樹未提交。

## 6b. B.3 閉環 frame targeting + NARROW_PINNED(chain 7)— P0-B 負面收案

B.3 依審計設計實作(`PARTNER_FRAME_CL`:`_layout_full` 上對 H 的 secant 閉環 ≤2 迭代 + hard snap,revert 保證;`PARTNER_COL_NARROW_PINNED`:narrow retry 綁 w_star 門控;13 新測試 + 87 guard 綠)。同鏈 4 臂 ×3:

| 臂 | paired Δ vs table | 95% CI | 分解 |
|---|---|---|---|
| +FRAME_CL | +0.0027 | [−0.0035,+0.0083] | area −0.0022、HPWL +0.0054 |
| +NARROW_PINNED | +0.0049 | [−0.0014,+0.0116] | HPWL +0.0137(!) |
| +兩者 | −0.0005 | [−0.0073,+0.0056] | v_rel −0.0010、area −0.0036、HPWL +0.0087;case 89 −0.0628 |

**P0-B 總結論:column arm 的 violation 修復已抵 Pareto 面** —— 本夜五個建構期修復(NARROW/GLUE/FRAME_CL/NARROW_PINNED/combo)全部呈同一簽名:v_rel/area 改善、SA 把釋放的鬆弛換回 wirelength、淨值 ≈ 0。約束面的移動只剩:時間(已收)、拓撲 prior(TFDL)、模型臂——後兩者卡舊機轉移。全部 flag 維持 default off,程式碼依 no-go 保留政策入庫。

## 7b. Proxy 校準稽核(deep-reasoner,100 案 ×2 reps,2,400 候選/rep)

- **Proxy 本體已是 gap 校準**:IEEE Access 2026 的 ~2% raw-magnitude 失準在我們的池上重現(raw HPWL 排序 regret +1.7~2.0%),但 gap 形式 proxy 的實際 regret 只有 **+0.16~0.18%**(已入袋)。`area_ref`(0.97 util 假設)實證精確(中位 1.001),不動。
- 發現的主導缺陷:跨 restart 仲裁吃容忍性 V 計數器(單向低估,142/2,400 候選)。R1(exact-V 仲裁 + 健全 early-exit,`PARTNER_PSEL_EXACT_V`)已實作;**paired ×4 端到端量測:+0.0052 [+0.0013,+0.0093] 顯著更差 → 不採,flag off**。候選層 regret 回收被管線效應反轉——本季教訓再演:candidate 指標不預測端到端。
- 1.5% 切換 margin:模擬顯示每一檔非零 margin 都更差(dead zone 擋掉的 swap 92% 官方正確,成本 ~0.003)——**但**先修 R2(direct 勝者「先計分後 `_ensure_no_overlap`」的不對稱,`PARTNER_PICK_SCORE_REPAIRED` 已實作、arms off 下休眠)再談改 margin;兩者等模型臂回歸後量測。
- 工件:`scratchpad/psel_audit/`(harness + pools ×2 reps)。

## 7c. Fast-SA 排程(dormant flag 首次 gate)與 arms 回歸(08-22 凌晨)

- `PARTNER_FASTSA_TEMP=1`(column-only,table 預算)7 對:**−0.0092,CI [−0.0201,+0.0031]**,HPWL −0.0227、area +0.0064;B 臂 σ 0.0064(高變異:case 89 −0.103、case 90 +0.082)。
- 新旗標 `PARTNER_FASTSA_MIX=<frac>`(per-config 排程多樣性入 restart portfolio,16 測試綠):arms-ON 下 +0.0019 [−0.0059,+0.0111],wash(HPWL −0.012、v_rel +0.0036)。**FASTSA 家族擱置,default off。**
- **舊機資產到位(08-22 01:38)**:LFS 15G、`eval_step1p2M.pt`、G0 語料、groupbridge 真身(兩顆 ckpt 解至 `submission/cadc1013/checkpoints/`)。
- **Arms-ON 基線(3D/3F + table 預算,GPU3)= 1.1676 ± 0.0072 @ avg 0.438s;模型臂價值 = −0.105(vs column-only 1.2726,CI [−0.138,−0.073])**;weighted HPWL gap 0.114 回到舊機水準。
- `artifacts/icdc_topology/tracer_step1/` 含一個 2,900-record 先導學生(contract hashes 合法)→ Task 5 pipeline 已驗通;正式 Task 5 執行中(deep-reasoner,GPU3)。

## 7d. Arms-ON 旗標歸因(chain 10,3 臂 ×3,箱內噪音升高 σ≈0.01)

| 臂 | paired Δ vs table+arms | 95% CI | 判定 |
|---|---|---|---|
| +PICK_EXACT_V | +0.0096 | [−0.0014,+0.0225] | off(exact 計數器家族端到端全負,與 PSEL_EXACT_V 一致) |
| +PICK_SCORE_REPAIRED | +0.0085 | [−0.0006,+0.0180] | off;margin 調整連帶擱置 |
| canonical 曲線(無 table) | **+0.0282** | [+0.0135,+0.0476] | **table 在 arms 下更強**;hidden 期望全情境 −0.016~−0.028(canonical+arms 同樣超 floor,quality 差距主導) |

Arms-ON table 平均 runtime 0.441s(canonical+arms 0.270s);M=2.1 下 58 案超 floor 點。arms-aware EV 表(2 檔位,17 格改 canonical)另存 `budget_table_arms.txt`;chain 11 同鏈 ×3:本地 +0.0050 [−0.0084,+0.0187]、省 0.037s/case —— 符合「以 quality 換 rt_adj」預期,淨效依賴 M/D 模型 → **預設維持 table v1,arms-aware 版為打包期依最終 runtime 校準決定的選項**。

- Arms-ON pseudo-hidden gate 基線(121 案 MIB 過濾,table env):**1.2157,121/121 feasible,avg 0.41s**;vs official100 arms ≈1.17 → gap +0.046(column-only 時 +0.048,一致)。

## 7e. TFDL Task 5 執行記錄(08-22,deep-reasoner)

- **Provenance 還原**:train-split 標籤來自 `scripts/probes/icdc_fp_teacher_g0.py --population-split train --sample-mod S --emit-corpus`(commit `9379002`,綁密封 G0 manifest `30902df2…`);heldout 語料由 `export_corpus.py` 自密封 G0 labels 導出(hash `b4ebef4d…` 逐位元相同)。`tracer_step1` 為 89 秒 trainer smoke,非正式跑。
- **規模裁定**:計畫未授權語料大小;全規模(sample_mod=1,~190k rows)投影 94 小時 → 採 `--sample-mod 32`:5,911 rows,2.4 小時,`TRAINING_LABELS_COMPLETE`,delta_h 0.2287(密封 G0 0.1858)。
- 所有 sha256 身分閘門 PASS(source/C0/flow/scorer/QA/G0 manifest;`source_ema = control_model = control_ema = 92838740…`);floorset_lite HF 重下載 receipts 逐位元吻合密封值(25 shard tracer,98/98 rows)。
- 順帶修復:`submission/cadc1013/{op_wrapper,op_src}.py` 遷移時漏傳,自 LFS tar 還原,hash 精確符合凍結常數;`--device cuda` 的 EMA device 錯位(`partner/icdc/train_topology_prior.py` 5 行修正,CPU 無作用;7 測試綠)。
- **阻塞與修正(scheduler 裁定,明確計畫修正)**:凍結 CLI 的 `--batch 4` 與 Task-3 的 padding 守衛(`topology_prior.py:620` 拒絕 energy.py 零填充 rects)衝突,任何 >1 batch 必敗;`--batch 1` 端到端可跑(推定即 tracer 先例)。**採 `--batch 1`,其餘參數不變**;方案 (B)(放寬守衛跑 batch 4)留待 G1 有望時審查。
- Per-case teacher 結果不可逐位元重現(wall-clock deadline 多執行緒 solver),聚合值一致(probe base_h 1.3468/delta_h 0.2039 vs 密封 1.3259/0.1858)。
- **訓練結果(`--batch 1`,5000 步,35 分,GPU3)**:best.pt @ step 5000,heldout topology loss 0.024764 → **0.022216(−10.3%)**,20 次 eval 全單調;separation −11.5%、contact −9.8%;spread 0.383→0.26→0.279(不塌縮);anchor ~2.6e-8(同形、貼近基底)。`checkpoint_contract.ok=true`。sha256 `4231d70e…`。曲線在 5000 步仍下降 → 步數預算為綁定限制。
- **Task 6(G1)正式儀式阻塞 — 三個工具缺陷(非候選證據)**:①`audit_topology_prior._decoded_candidates` 拿未合法化 raw DPM++ decode(student_cost 飽和於 10.0)比密封標籤中經完整產線合法化的 base/teacher cost → retained_gain −31.3;**null control(與 control 逐位元相同權重)同樣 −31.7** → 任何 checkpoint 都不可能過;②`causal_smoke.compare_smoke_arms` 要求兩臂 Flow hash 相等,但 `_sample_arm` 未 pin Flow RNG → 不變量不可能成立;③計畫要求的 freeze/blind-pair runner/adjudicator(`run_icdc_topology_stage.sh`、`adjudicate_g1`、`freeze_stage_checkpoint` 等)**從未存在於任何 commit**;④既有 `compare_g1_arms` 用含 runtime 的 weighted_combined、零 margin。Handoff 所稱「sealed G1 evidence tooling 已實作」只對 evidence primitives 成立。
- **Scheduler 裁定:G1-lite**。同 session 交錯 ×3 full-100,control=`direct_v2_final.pt` vs candidate=`best.pt`(chmod 0444,sha `4231d70e…`)作 `DIRECT_CKPT`,其餘凍結 P_B_3D3F env(canonical 預算、無實驗旗標、3D/3F)—— 實質等同 G1 因果實驗,省去 freeze 儀式。通過線:candidate ≤ control − 0.015(paired,CI 支持)。正式儀式層若日後需要,先修 ①②再建 ③。
- **G1-lite 結果(v1 學生,3 對)**:control 1.1985 ± 0.0060 vs student 1.1898 ± 0.0060;**paired −0.0086 [−0.0228,+0.0061]**,n≥100 band −0.0095;分解 v_rel −0.0032(拓撲 prior 降違規)、area −0.0023、HPWL 持平;大勝 92(−0.104)/94(−0.081),反例 83/85(+0.08/+0.07)。未達線但方向正確;學生只訓 5000 步且曲線未收斂 → 同語料 20000 步 v2 訓練中。**6 對合計:−0.0086 [−0.0204,+0.0048],5/6 勝,估計值與 3 對時相同(穩定);分解 v_rel −0.0024、HPWL −0.0076、area +0.0023。** v1 效應 ≈ −0.009,為 −0.015 線之半。在 table 預算(出貨組態)下 3 對:−0.0045 [−0.0176,+0.0083],v_rel −0.0027、HPWL −0.0064、area +0.0101(噪音高,σ 0.015)。
- **學生 v2(同語料,20000 步,batch 1,eval/1000)**:best @ step 18000,heldout topology loss 0.024764 → **0.020963(−15.3%)**,20k 前收斂(best ≠ final);sha `c739c78c…`;`checkpoints_s2_20k/`。**G1-lite v2(凍結 env,3 對):control 1.2129 ± 0.0111 vs v2 1.1970 ± 0.0048;paired −0.0159 [−0.0287,−0.0016],3/3 勝,n≥100 band −0.0164;分解 HPWL −0.0225、v_rel −0.0016、area −0.0015;runtime 0.283→0.269。→ 通過 −0.015 相對線(CI 排除 0)= HIGH_TAIL_CAUSAL_PROOF 等價;依計畫 STOP_REQUIRES_SEPARATE_APPROVAL(G2/打包需人工核准)。** 後續(實驗層):table 預算確認 ×3 + 凍結 env 再 3 對(chain 15)。
- **Chain 15 降溫(白天高負載,arm σ 0.015)**:凍結 env 6 對合計 **−0.0079 [−0.0200,+0.0039]**(r4–6 變弱);table 預算 3 對 −0.0021 [−0.0193,+0.0164](HPWL −0.0261、v_rel +0.0046)。**穩健結論:v2 的 HPWL 增益真實(−0.020~−0.026 各組一致),淨效在 pooled 證據下未達 −0.015 線;v2 = 有望未證。** 下一步 v3(sample_mod 16 語料 ≈2× rows,20k 步)+ 安靜時段 6 對重量。
- **學生 v3**(teacher sample_mod 16:11,925 rows,4.8h;20k 步):best @ 20000(仍在降),heldout loss 0.024764 → **0.020805(−16.0%)**;sha `50d43ef1…`;`checkpoints_s3_m16_20k/`。安靜窗口(load 9–13)三臂交錯 ×4(chain 16)量測中。
- **Chain 16(安靜窗口,arm σ 0.002–0.004,凍結 env ×4)**:control 1.1934 ± 0.0036;**v2 1.1851 ± 0.0020,paired −0.0083 [−0.0173,+0.0024],4/4 勝**(HPWL −0.0240、area +0.0045、v_rel +0.0008;n≥100 −0.0100);v3 1.1907 ± 0.0104,−0.0027 [−0.0157,+0.0101](proxy loss 更好但下游更差,語料加倍無益)。**結論:v2 真實效應 ≈ −0.008(三組獨立量測一致:−0.0159/3、−0.0079/6、−0.0083/4),未達凍結計畫 −0.015 線;作為零 runtime 代價的候選仍有出貨價值,依 table 預算安靜量測(chain 17)決定建議。**
- **Chain 17(table 預算,安靜窗口 ×4)**:control 1.1668 ± 0.0075 vs **v2 1.1608 ± 0.0060,paired −0.0060 [−0.0167,+0.0049]**,3/4 勝;HPWL −0.0199、v_rel +0.0020、area −0.0003;runtime 持平(0.431 vs 0.429)。
- **打包建議(需使用者核准,計畫 STOP 條款)**:下一版包採 `PARTNER_BUDGET_TABLE`(本機 −0.019~−0.028,hidden 期望 −0.013~−0.028)+ v2 學生作 `DIRECT_CKPT`(−0.006~−0.008,零 runtime 代價;HPWL −0.02 穩健,被 v_rel +0.002 部分回吃)。**本機出貨組態最佳值 1.1608**(同箱 canonical+arms control 1.19–1.21)。v3 擱置(語料加倍無下游增益)。

## 9b. Arms 時代 env 槓桿(chain 18–19,v2+table 基線 1.1608,安靜窗口)

| 旗標 | paired Δ | 95% CI | 判定 |
|---|---|---|---|
| `PARTNER_DIRECT_SEAT_FIX=1`(canonical 預設 0) | −0.0098 | [−0.0201,+0.0015] | 採 |
| `PARTNER_NREF=9`(預設 6) | −0.0108 | [−0.0243,+0.0035];+0.011s | 採 |
| **兩者合併**(4 對) | **−0.0107** | [−0.0236,+0.0014],4/4 勝 | **promote → 1.1512** |
| `PARTNER_PICK_MARGIN=1.0`(新 env,預設 0.985) | 單獨 −0.0035;疊加 sn 上 +0.0025 | — | 維持 0.985(稽核的 column-only 模擬未轉移到 arms) |
| `PARTNER_FASTSA_TEMP=1`(arms) | +0.0022 | — | off(終判) |
| table v2(4 檔資料) | 0.0000 | — | 維持 v1 |

Chain 20(sn 基線 1.1485,3 對):`PARTNER_NREF=12` −0.0004(9 為膝點);`PARTNER_OVERSAMPLE=2` **+0.0087 [+0.0003,+0.0188] 更差**(候選越多越差再證);`PARTNER_FLOW_SLOTS=5`:**7 對 −0.0145,arm 1.1511 ± 0.0061 → 1.1366 ± 0.0058**,n≥100 −0.0156,v_rel −0.0053、area −0.0124、HPWL +0.0094;case-bootstrap CI [−0.0404,+0.0037] 因少數巨案擺動偏寬,arm 層 2.4σ 分離 → **採納**(+0.002s)。

Chain 21(fl5 基線 1.1335,3–4 對):`FLOW_SLOTS=7` +0.0054、`KS_CAP=8` +0.0053、`FLOW_STEPS=12` **+0.0132 [+0.0010,+0.0270]** → 全部否決,供給軸停在 FLOW_SLOTS=5。

**出貨組態(本機最佳 1.1366)**:3D/**5F** + table + v2 + SEAT_FIX + NREF=9。:3D/3F + `PARTNER_BUDGET_TABLE`(v1)+ v2 學生 DIRECT_CKPT + `PARTNER_DIRECT_SEAT_FIX=1` + `PARTNER_NREF=9`。同箱 canonical+arms control ≈1.20 → 累計 −0.05。

## 10. 總結(08-22 14:00 UTC;9b 為其後增補)

| 項目 | 本機證據 | 狀態 |
|---|---|---|
| PARTNER_BUDGET_TABLE | column-only −0.019;arms −0.028;hidden 期望全情境勝 | **promote** |
| TFDL 學生 v2 作 DIRECT_CKPT | −0.006~−0.008(4 組量測一致),HPWL −0.02 | **promote 候選**(需核准) |
| ePlace arm | 合法化後 21/21 敗 | kill |
| NARROW/GLUE/FRAME_CL/NARROW_PINNED | Pareto wash | off |
| FASTSA_TEMP/MIX | column-only 趨勢正、arms wash | off |
| PSEL_EXACT_V/PICK_EXACT_V/SCORE_REPAIRED | 端到端負 | off |
| v3 學生(2× 語料) | −0.0027 | park |

## 8. 阻塞(需人工)— **已解除(08-22 01:38 轉移完成)**

舊機 `/nashome/NVL4/vdalab/yyds-dev/Data-Driven-SoC-Floorplanning/` 最小轉移集(~18GB):
`.git/lfs/objects/`(15GB)、`partner/checkpoints/direct_v2_cont/eval_step1p2M.pt`、`artifacts/icdc_topology/`、`artifacts/icdc_g0_v2_area_full/`。(floorset_lite 已 HF 還原,不必轉。)

## 9. 補記(chain 5 確認鏈)— **PARTNER_BUDGET_TABLE 本機促轉**

base/table/b1v2 ×3 同鏈交錯:

| 臂 | noRT(3-rep) | avg_rt | paired Δ vs base | 95% CI |
|---|---|---|---|---|
| base(canonical 曲線) | 1.2908 ± 0.0026 | 0.224s | — | — |
| **table(EV 表)** | **1.2715 ± 0.0035** | **0.384s** | **−0.0193** | **[−0.0321, −0.0088]** |
| b1v2 | 1.2683 ± 0.0017 | 0.483s | −0.0225 | — |

- Table vs b1v2:本地 +0.0032 [−0.0064,+0.0151],但省 0.099s/case;依 §2 情境模型,hidden 期望 table 優於 b1v2(b1v2 在 M≥2.1 情境超 floor 受罰),且最壞角落 table ≈ base、b1v2 −0.030。**採 table。**
- Table 增益分解:HPWL −0.0139、area −0.0068、v_rel −0.0028;n≥100 band −0.0120。
- 大樣本 pseudo-hidden(5-per-n,MIB 過濾後 121 案,table env):**1.3198,100% feasible** —— 立為 promotion gate 的 pseudo 基線。
- 本夜本機(column-only)口徑合計:**1.2900 → 1.2715(−0.019)**;hidden 期望(含 rt_adj)≈ −0.013,最壞情境 ≈ 0。
- 注意:chain 5 的 tblA_r1 因共機 22:30–23:00 負載尖峰跑了 35 分(solver avg_rt 正常 0.225s)——絕對 runtime 在本機不可信,促轉依據全部為同鏈相對值。

## 11. shadow_hidden 四套壓力測(08-26,shipping 組態 = table + v2 + SEAT_FIX + NREF9 + FLOW5,GPU3)

| 套 | 性質 | noRT | feas | avg/p90/max rt | hpwl | area | v_rel | bnd/grp/mib |
|---|---|---|---|---|---|---|---|---|
| v1 | 訓練 heldout 100(MIB 偏差) | 1.3453 | 100 | 0.45/1.04/1.42 | 0.119 | 0.091 | 0.096 | 1.73/0.97/3.73 |
| v3 | 低 MIB 校準版 | 1.2111 | 100 | 0.46/1.04/1.49 | 0.135 | 0.074 | 0.045 | 1.97/0.53/0.28 |
| alpha_1 | official100,P2B 端點按距離重抽 | 1.2117 | 100 | 0.41/0.88/1.49 | 0.202 | 0.065 | 0.032 | 1.40/0.49/0 |
| p2b_1__b2b_1 | P2B+B2B 皆重抽 | 1.1052 | 100 | 0.40/0.81/1.21 | −0.049 | 0.049 | 0.036 | 1.68/0.47/0 |

讀法:四套皆 100/100 feasible、runtime 尾 ≤1.5s、bnd/grp 與 official100 同量級 → 分佈偏移不崩。v1 差額幾乎全是 MIB 工件;v3 +0.074(boundary 2.0/案 + area)。alpha_1 HPWL gap 翻倍(golden 基準因「pin 靠近 GT block」重抽而偏有利 GT)→ pin 驅動拓撲是最弱項。p2b+b2b 重抽後 golden 基準變鬆,HPWL gap 為負。工件:`artifacts/shadow/sh_*.json`;runner `run_shadow.sh`(job scratch)。

Beta 時代組態(canonical 曲線 + 原 direct ckpt + FLOW3)對照(同箱同日):

| 套 | beta 組態 | shipping 組態 | Δ | 主要來源 |
|---|---|---|---|---|
| v3 | 1.2914(hpwl .179/area .113/v .058) | 1.2111 | **−0.080** | area −0.039、hpwl −0.044、v_rel −0.013 |
| alpha_1 | 1.2307 | 1.2117 | −0.019 | hpwl −0.017 |
| p2b+b2b | 1.1067 | 1.1052 | −0.001 | 已貼品質下限 |

→ shipping 組態在偏移資料(v3)上的增益(−0.080)**大於** official100(−0.06):預算表/refine 容量的改善泛化到未見 instance。工件 `artifacts/shadow/shb_*.json`。

alpha_1 診斷:HPWL gap 隨 n 增大(n 90–109:0.254 vs official 0.136;110–120:0.177 vs 0.074),最慘案(90/74/110/50/106/76)自 gap≈0 跳至 0.3–0.4;p2b 佔 HPWL 19–27%。內部 proxy `_hpwl`(column_sa_legalizer.py:2477)與 evaluator 逐項同形 → 非目標失準,是「pin-heavy block 需放到 pin 側」的拓撲搜尋弱項(seed 通道已死,需新 SA move/prior)。定位為下一階段靶,alpha_1 作為診斷 gate(非 hidden 代理)。

## 12. 官方 0823 更新(partner handover 附件)+ partner handover 對照(08-26)

- **Leaderboard 更新**(`docs/official/beta_test/C_beta_leaderboard_update_20260823.csv`):我們 raw 1.3207 不變、total 0.9266 → **第 4**(rank1 raw 改為 1.0845;rank5 raw 1.1705 但 133s 被罰;rank9 raw 1.1242/83s)。我們 total/raw=0.7016 仍貼 floor。
- **Median 表更新**(`C_median_runtimes_beta_hidden_update_20260823.csv`):總和 295.7→216.1s(−27%,單案比 0.48–0.94)→ floor 點縮小。舊 EV 表在 arms-ON 下 70 案超新 floor;依新 median 重推的表(`budget_table_newmed.txt`,54 案回 canonical,平均 0.426→0.326s)離線期望 −0.008(vs 舊表)/−0.019(vs canonical);M=1.8 情境算出 0.9267 ≈ beta 實際 0.9266 → M 校準中心改 1.8。三套 gate 確認中。
- **Partner handover v1**(`shadow_hidden/handover_v1.tar.gz`,scratch 解至 job tmp):與本線收斂(runtime headroom 為第一槓桿;pool 40/mix 更差;ePlace 只 probe)。可驗證分歧:①拿掉 TAG_COMPRESS/GROUP_BRIDGE 反而 −0.028(2/2)②Flow-only(FLOW_SLOTS=10)優於 3D/3F ③coord_polish 的 headroom gate(−0.0026 noRT/−0.037 runtime-aware)④postpass router(v3 −0.0024)⑤MIB decouple(8/24)。①②以三套 gate 在本線驗證中;③④⑤待 diff 抽取後決定移植。其 public 1.1356 / v3 1.1987 vs 本線 1.137 / 1.211。

Chain 22(sn+fl5 基線,3 對):`PARTNER_POOL=32` **−0.0055 [−0.0107,+0.0004]**(runtime 中性;partner 測 40 反而 +0.016 → 32 為甜蜜點,待三套 gate 確認);`PARTNER_NREF_MIN_N=80` −0.0031 [−0.0097,+0.0032] wash。

### 12a. 更正:我們的 beta 列 = raw 1.3141 / 38.15s(非 1.3207/52.07s)

- 原榜第 7、**更新榜第 8(total 0.9795)**;total/raw = **0.7454 → 新 median 下我們已不在 floor 上**(runtime 罰分 ≈ +6.5%)。前文以 1.3207 列推導的「全案貼 floor」結論僅對舊 median/舊列成立;raw 賽局位置不變(第 7–8)。
- 本機→contest 時間係數以真實列反推:canonical+arms 組態(本機 avg 0.270s)重現 0.7454 ⇒ **M = 1.45**(取代先前 1.8–2.1 的估計)。
- 在 M=1.45 + 新 median 下的 hidden 期望 total(含 rt_adj):canonical 0.8975、舊 EV 表 0.8778(60 案超 floor 但罰分淺)、**校準表 `budget_table_cal.txt`(64 table/36 canonical,avg 0.39s)0.8738**。三套 gate 驗證中(gT vs gCal)。對照更新榜 rank1 0.8586 / rank2 0.8882。

### 12b. 「決賽 field 只會更快」原則下的預算表(user 裁定)

- 情境先驗改為 D∈{0.7:0.30, 0.8:0.35, 0.9:0.20, 1.0:0.15}(相對更新版 median)× M=1.45±15%,重推 **`budget_table_fast.txt`**(54 table/46 canonical,avg 0.351s)。hidden 期望(public per-case 資料,M=1.45):

| D | canonical | 舊表 | cal 表 | **fast 表** |
|---|---|---|---|---|
| 0.6 | 1.0107 | 1.0033 | 0.9982 | **0.9941** |
| 0.8 | 0.9434 | 0.9284 | 0.9245 | **0.9219** |
| 1.0 | 0.8975 | 0.8778 | 0.8752 | **0.8742** |

- **Public 不是 hidden 代理**(beta hidden 1.314 vs public 1.16),絕對值改用 v3/alpha_1 per-case 資料估:v3 上 shipping vs canonical 在 D=0.7/0.8/1.0 皆 −0.038~−0.049(D=0.8:0.964 vs 1.006);alpha_1 因品質增益僅 −0.019,D≤0.8 時多花時間反成 +0.005~+0.008 → 預算只值得花在品質增益大的帶,fast 表即此折衷。三套 gate(舊表 vs fast 表)驗證中。

### 12c. alpha_1 診斷更正(deep-reasoner 通道隔離實驗)+ 新 lever `PARTNER_COL_BALANCE`

- **更正**:alpha_1 的 HPWL 翻倍 **94% 來自 b2b、僅 6% 來自 p2b**(我們 p2b 已在 golden 的 +5% 內)。把 official 版佈局改用 alpha_1 的 p2b 計分,gap 只有 0.114;隔離實驗(只把模型條件用的 p2b 換回 official,其餘全 alpha)精確重現 official 結果 → 真因 = **direct/flow 模型對 p2b 分佈偏移脆弱,管線退回 column SA**;而 column SA 單獨的 b2b 是 golden 的 1.4–2.0×,在 official100 最差幾案(79/47/80/62)本來就在承重。seed 換 p2b、SA 目標換/去 p2b 皆無效(seed 通道死、目標非因)。alpha_1 應視為「column fallback 地板」的 gate,非 hidden 風險訊號。
- **建構期根因**:`_init_columns`(column_sa_legalizer.py:1181)單向游標 + 等面積 cap + rigid 守衛永久跳過 column → 空 column、末 column 雙倍負載,初始寬度 1.39× 估計(tid 69:152.6 vs golden 142)。DP 等「寬度」連續分割(目標 Σ_c max(A_c/(H−R_c), maxrigid_w_c))離線 10 案:中位 −10%,8/10 ≤ golden 寬。**`PARTNER_COL_BALANCE` 實作中**(O(C·m²) 向量化,~2ms)。另:`C0=round(W_est/√avg_area)` 系統性偏低 2。
- `hp_ref` 以首佈局自舉(SA 低估 HPWL 1.4×)實測修正在噪音內(anneal 自身重校已覆蓋),不動。
- **「學 slicing tree」訓練假說否決**:official100 golden 佈局 **0/100 可 guillotine 分解**(連 n=21 都無任何貫穿切線)→ golden 非 slicing 結構;若要走「模仿 golden 拓撲」的生成模型,表示法得是 sequence pair / B*-tree(週級工程、高風險)。

## 13. 打包前置(P0-A 準備,08-26)

- Python 3.13 由 `uv python install 3.13` 取得(系統無);乾淨 venv + requirements 單案跑通(官方 evaluator、PYTHONPATH 只給 FloorSet/)。解析版本:torch 2.13 / numpy 2.5 / numba 0.67 / llvmlite 0.49。
- requirements.txt 草稿(job scratch `pkg/requirements.txt`):`torch>=2.5.0 numpy>=1.26.4 numba>=0.61.0 tqdm>=4.66.4 scipy>=1.13.0`(scipy 為 coord_polish 依賴,缺則靜默退化)。
- 待打包日確認:`synth_instances` 模組入包;解壓 tar 以 debug 探針確認 `loaded direct model` / `loaded flow model`;stdlib 3.13 相容無問題。

## 14. 三套 gate 主鏈(08-26,load 60–80 高噪音,2 reps,判準 v3+alpha_1 為主)

| 組態 | official | v3 | alpha_1 | mean | 判定 |
|---|---|---|---|---|---|
| A shipping(table+v2+SEAT+NREF9+FLOW5) | 1.152 | 1.225 | 1.235 | 1.204 | 基線 |
| B 去 TAG_COMPRESS+GROUP_BRIDGE | 1.147 | 1.218 | 1.230 | 1.198 | 三套皆 −0.005~−0.007 → 採(partner 的 −0.028 未重現,方向成立) |
| C Flow-only(FLOW_SLOTS=10,Direct 0 席) | 1.140 | 1.206 | 1.235 | 1.194 | off/v3 −0.012/−0.018,a1 持平 → 採(待 B+C 合併確認) |
| F fast 預算表 | 1.150 | 1.242 | 1.228 | 1.207 | +0.003 = 少花時間的品質代價;依「field 變快」原則採 fast 表 |
| POOL 32(1 rep) | 1.156 | 1.249 | 1.247 | 1.217 | 高 load 污染,不採 |

| POOL 32(2 reps) | 1.156 | 1.237 | 1.242 | 1.212 | 否決 |
| partner bundle 原值(polish headroom 1.0 / router / flow warm) | 1.149 | 1.225 | 1.228 | 1.201 | −0.003(a1 −0.007);微調鏈重測門檻 |
| v3 建構旗標三合一(FRAME_CL+NARROW_PINNED+GLUE,v3 ×2) | — | 1.2223 vs base 1.2209 | — | — | wash,徹底判死 |

B+C+fast 合併確認鏈排隊中(comboBC / comboB / comboC ×2)。

- `tests/test_partner_frame_scale.py::test_off_path_is_bit_exact{,_with_preplaced}` 在高 load 下失敗(同臂兩跑即不同);HEAD worktree 二分:**HEAD 本身亦非確定** → 與今日編輯無關,為時序敏感 flake(refine 路徑內仍有 wall-clock 依賴)。非打包阻塞;守則:打包驗證用「解壓 tar 官方指令 100/100 feasible + 分數帶內」,不以 bit-exact 為門檻。
- `PARTNER_COL_BALANCE` 落地(7 測試綠;100 案建構檢查:寬度中位 **−9.1%**,零案變差);三套 gate 排隊(組合鏈後)。

## 15. 08-26 下午鏈驗收(接手 session;所有判定只用同鏈交錯相對值)

先前 session 排隊的四條鏈全部跑完但未記錄;同鏈內比較如下(三套 = official / v3 / alpha_1,mean 為三套平均):

| 鏈(fast 表 + v2 + SEAT_FIX + NREF9) | r1 mean | r2 mean | 判定 |
|---|---|---|---|
| comboC = FLOW_SLOTS=10(Direct 0 席),TAG_COMPRESS=1/GROUP_BRIDGE=1 | 1.1970 | 1.1951 | **採(最佳)**;official 1.1349/1.1293 |
| comboBC = C + 去 TAG/BRIDGE | 1.2017 | 1.2006 | official 比 comboC 差 +0.014/+0.020;v3/a1 持平 → **B 疊在 C 上有害,保留 TAG_COMPRESS/GROUP_BRIDGE** |
| comboB = FLOW5 + 去 TAG/BRIDGE | 1.2111 | 1.2072 | 與 tuneBase(FLOW5,1.2095)持平 → B 單獨 wash;§14 的「B 採」撤銷 |
| cbBase(=comboBC)vs cbBal(+COL_BALANCE) | 1.1991 → 1.1976 | 1.1953 → 1.1959 | wash;安靜窗口再驗(qC_r1 vs qCbal_r1,C 組態):off +0.0038、v3 +0.0092 [+0.0003,+0.0190]、a1 +0.0099 → **COL_BALANCE 否決**(寬度 −9% 是真的,但 SA 把它換成 HPWL/area 付回) |
| v4gV2 vs v4gV4(aug 學生 `checkpoints_s4_aug`,FLOW5+B 組態) | 1.1948 vs 1.1935 | 1.1947 vs 1.2033 | v4 official +0.005/+0.018 → **v2 維持**;且在 C 組態下 Direct 0 席,DIRECT_CKPT 只剩「必須載入才開臂」的角色,TFDL 學生通道實質休眠 |
| tune 鏈(partner bundle 微調:POLISH_HEADROOM 1.5/2.0、ROUTE_HEADROOM 0.6/1.2) | 1.2088/1.2060/1.2075/1.2031 vs base 1.2055 | r2 未跑完 | 全在噪音內,不採 |

跨鏈絕對值不可比:同一組態(FLOW5+B)在 load 60–80 時 official 1.1647/1.1495(comboB),在 load 20 時 1.1375/1.1352(v4gV2)——差 0.02 全是 load(SA 吞吐)。

**目前最佳出貨候選 = C**:fast 表 + v2 DIRECT_CKPT + `PARTNER_DIRECT_SEAT_FIX=1` + `PARTNER_NREF=9` + `PARTNER_FLOW_SLOTS=10` + TAG_COMPRESS=1 + GROUP_BRIDGE=1。安靜窗口(load 7–15)qC_r1:**official 1.1333 / v3 1.2073 / a1 1.2157**(mean 1.1854);runtime-aware 期望(M=1.45,新 median):D=0.7 0.915、D=0.8 0.885、D=1.0 0.840。

### 15a. 中段饑餓懸崖的真因 = 模型臂開臂閘門(不是 SA 時間)

comboC official 逐案(§14 組態,load 60):n≥102 的案 hpwl gap 0.00–0.08、cost 1.00–1.18;n=83–101 中拿 canonical 預算(0.05–0.38s)的 16 案 hpwl gap 0.18–0.63、**平均 cost 1.317**,同帶拿 table 預算(0.53–0.80s)的 10 案平均 **1.166**。這不是 n 的函數而是預算的階躍:

- `_direct_seat_ok`(contest_optimizer.py:236–263,`PARTNER_DIRECT_SEAT_FIX=1`)只在 `0.7·(0.85·budget − 0.148·n/100) ≥ 0.125·(n/100)²` 時保留 refine 席位 → 開臂預算 ≈ 0.25s@n76、0.33@n90、0.38@n100、0.39@n101;fast 表在 76–101 有 16 個 n 低於此線 → 整案退回 column SA(column-only 資料同帶 cost 1.2–1.5,和 arms 開臂後 1.05–1.17 差 0.15–0.25/案)。
- 舊 EV/fast 表是用 column-only 三檔品質資料推的,看不到這個臂的階躍(column-only 在 n=100 從 0.32s→0.73s 只值 −0.03,arms 開臂值 −0.23)。
- 帶內加權殘量:n<76 0.006、**76–101 0.045**、102–120 0.077–0.087(權重 0.023/0.182/0.795)。把 16 個 canonical 案拉到 1.17 ≈ **−0.017~−0.02 raw**。

`artifacts/p0_newbox/budget_table_mid.txt` = fast 表,但 n∈[76,101] 取 max(fast, 開臂線+0.07s)(0.324@76 … 0.460@101;16 個 n 改動,平均預算 0.351→0.386s)。runtime 面:這些值在 D≥0.8 全在 0.3006·median/1.45 免費區內;D=0.7 時 n=78–87 超出 5–15%(單案 rt 罰 +3~5%,遠小於 −12% 的品質增益)。鏈 qMid vs qC ×3(GPU3)量測中。

### 15b. 進行中

- partner 的 **quota-first 候選生成**(Flow 佔滿席位時完全不跑 Direct 採樣;Flow 失敗時 Direct 回補)本線未移植 —— C 組態下每案仍白付一批 Direct dpmpp(算在 0.148s 的 sampler 延遲內,直接抬高開臂線;contest A100 上更貴)。worktree 移植 + bit-exact 驗證中。

### 15c. 中段開臂表(budget_table_mid)gate 結果 — **promote**

同鏈交錯 qMid ×3 vs qC ×4(C 組態,GPU3,load 15–25):

| 套 | fast 表(qC,4 reps) | mid 表(qMid,3 reps) | paired Δ | 95% CI | 分解 |
|---|---|---|---|---|---|
| official | 1.1356 ± 0.0034 | **1.1243 ± 0.0071**(r2 1.1180) | **−0.0113** | [−0.0240, −0.0014] | HPWL −0.028、area +0.002、v_rel +0.001;n≥100 帶 −0.0006(預算相同,純噪音) |
| v3 | 1.2028 ± 0.0040 | **1.1844 ± 0.0058** | **−0.0183** | [−0.0450, +0.0008] | HPWL −0.035 |
| alpha_1 | 1.2202 ± 0.0037 | 1.2151 ± 0.0018 | −0.0051 | [−0.0155, +0.0043] | a1 = column fallback 地板,臂開了也難贏,符合 §12c |

- 三套 mean 1.1854/1.1856/1.1869 → 1.1744/1.1746/1.1748,**3/3 rep 同向**;avg runtime +0.045s/case(0.372→0.416),max 持平。
- Runtime-aware(M=1.45、新 median,official per-case):D=0.7 0.9196→0.9115、D=0.8 0.8893→0.8792、D=1.0 0.8435→0.8338 —— 悲觀情境也贏 −0.008~−0.010(超 floor 案 46→57,但罰分淺)。
- 增益機制逐案驗證:tid 55/57/58/62/64/66/68/69/73/78/80 從 hpwl 0.18–0.65 降到 0.00–0.08(69:1.571→1.127、62:1.291→1.114、73:1.219→1.057);n=96/100(tid 75/79)在 0.43/0.45s 仍未開臂(見 [seat] 診斷)。
- **判定:promote `PARTNER_BUDGET_TABLE` = budget_table_mid.txt**(v3+a1 mean −0.012,official −0.011,符合使用者判準)。出貨組態 = C + mid 表。

### 15d. `[seat]`/`[rp0]` 診斷跑(official 100,mid 表 + C,`PARTNER_SEAT_DEBUG=1`;本跑 noRT **1.1167**)

- 暖機後 Flow sampler 延遲 ts = 0.075–0.128s(n=76–120),校準常數 0.148 仍略保守;quota-first 單案 smoke 看到的 0.49s 是冷啟(首案)——contest 端首個開臂案會付一次,`PARTNER_FLOW_WARM=1` 可消(gPart 鏈中為 wash,打包時可開)。
- 中段開臂案的 refine 視窗 left = 0.18–0.29s,tail 0.26–1.11s;n≥76 全部拿到席位(62/62)。
- **rung-0 固定框成功(r0=1)極少**(tail 多為 0/9),但 expand rung 照樣交出 hpwl 0.00–0.07 的佈局 → r0 不是交付訊號。
- **殘餘 fallback 與時間無關**:tid 63(n=84,left 0.44s)、75(n=96,0.27)、79(n=100,0.28)、60(n=81,0.18)在 3/3 mid 表跑中全部 fallback(hpwl 0.24/0.42/0.49/0.06–0.24);tid 61(n=82,left 0.42)則穩定成功。instance 統計(fixed/preplaced/boundary/MIB/cluster 計數)看不出差異 → 需 `REFINER_DEBUG=1` 的 `[pick]` 看候選是否存在/分數(排隊在 chain F 後)。
- tail 隨機 fallback(9 跑:tid 86 2/9、97 4/9、98 3/9,單跑殘量 0–0.025、平均 ≈0.011)同樣待 `[pick]` 診斷。

### 15e. Chain F(mid 表 + C 基線,GPU3,load 18–21;r1 全臂 + r2 只留 QF/base)

| 臂 | official | v3 | alpha_1 | paired Δ vs base(off) | 分解(off) | 判定 |
|---|---|---|---|---|---|---|
| fBase(mid+C,quota-first 碼已入樹、toggle off)r1/r2 | **1.1161**/1.1234 | 1.1763/1.1807 | 1.2047/1.2171 | — | — | 基線(三套 mean 1.1657/1.1737) |
| `PARTNER_QUOTA_FIRST=1`(r1/r2) | 1.1219/1.1263 | 1.1745/1.1717 | 1.2207/1.2081 | 2 reps:off +0.0043 [−0.002,+0.011]、v3 −0.0054、a1 +0.0035 | — | **wash → default off**(碼保留;本機暖機 sampler 只 0.08–0.13s 無可省;contest A100 若冷啟延遲大可再開) |
| `PARTNER_FLOW_STEPS=4` | 1.1405 | 1.1678 | 1.2085 | +0.0244 | area +0.017、v +0.006 | **否決**(候選品質換不到延遲) |
| `PARTNER_FRAME_SCALE_SET=1.01` | 1.1287 | 1.1840 | 1.2187 | +0.0125 | HPWL +0.017、area **+0.005** | 否決:更緊的固定框只讓 rung-0 更常失敗,area 反而不降 |
| `PARTNER_FRAME_SCALE_SET=1.00` | 1.1300 | 1.1857 | 1.2130 | ≈+0.012 | — | 否決(0.3s 時代的 −0.018 在高預算/expand 路徑下不再成立) |
| ladder 1.01,1.02 | 未跑(r1 五臂皆負後截斷) | | | | | — |

結論:tail area gap ≈0.05 不能靠 rung-0 框常數回收(1.02 已是甜蜜點);Flow 步數 8 為下限。

### 15f. Fallback 真因(`REFINER_DEBUG=1` 全跑 → `[psel]` 池內仲裁記錄;單案跑不可用:每 process 首案 sampler 冷啟 0.57–0.63s,left<0)

pool 路徑的仲裁在 `column_sa_legalizer._parallel_solve`(4740–4760):`score=(1+0.5·((hp−hp_ref)/hp_ref+max(0,area/area_ref−1)))·exp(2V/n_soft)`,direct 需 < 0.985×column。n≥76 的 45 案:37 案 direct 勝、7 案 column 較佳、1 案 dead zone;**7 個 column 勝 = 全部 fallback 案**,且每一案 direct 候選都存在(3 個)、HPWL 低 10–25%,輸在 V:

| tid | n | column (hp, V, score) | best direct (hp, area vs col, V, score) | ratio |
|---|---|---|---|---|
| 79 | 100 | 157, V0, 1.2245 | 121, +7.6%, **V6**, 1.3855 | 1.131 |
| 75 | 96 | 160, V1, 1.2265 | 128, +18%, **V8**, 1.5264 | 1.245 |
| 64 | 85 | 198, V5, 1.4588 | 148, −3%, **V10**, 1.4730 | 1.010 |
| 63 | 84 | 135, V1, 1.1467 | 131, +2%, V3, 1.2443 | 1.085 |
| 67 | 88 | 70, V1, 1.1168 | 70, +9%, V2, 1.2133 | 1.086 |
| 60 | 81 | 114, V0, 1.0618 | 101, +16%, V2, 1.1846 | 1.116 |
| 65 | 86 | 133, V1, 1.2389 | 107, +2%, V4, 1.2360 | 0.998(dead zone) |

- 官方 cost 同形(exp(2V/n_soft)),所以仲裁**正確**——問題是中段 direct 候選的 V 沒修完:`refine_prediction` 的違規修復 reserve = `min(3.5, 0.3·slice)`,中段 slice 0.2–0.3s → 只有 0.06–0.09s;direct 候選在 ladder 內已 `_edge_seat` 過,殘餘 V 是 grouping/深層 boundary。tail(slice 0.5–1.1s)direct V=0–2 → 穩定勝。
- 08-22 的 proxy 稽核結論(仲裁 regret 0.16%)在此仍成立;槓桿不在仲裁而在候選修復時間。
- 下一鏈(chain G):①`budget_table_mid3`(16 個開臂 n 再 +0.12s,avg 0.386→0.405s)②新 knob `PARTNER_REFINE_RES_FRAC`(default 0.3 bit-exact;試 0.45)③兩者合併。

- `budget_table_mid2`(再開 n=60–75 的 7 個 n,avg 0.386→0.401s)×2:off +0.0040 [−0.005,+0.015]、v3 +0.0129 [+0.001,+0.025]、a1 −0.0021 → **否決**(n<76 權重只 2%,無可回收)。

### 15g. Chain G(mid 表 + C 基線)— **`PARTNER_REFINE_RES_FRAC=0.45` promote**

新 knob(layout_refiner.py `refine_prediction`:`res = min(3.5, RES_FRAC·slice)`,default 0.3 = 出貨常數 bit-exact):

| 臂 | official(r1/r2) | v3 | alpha_1 | paired Δ(2 reps)off / v3 / a1 | 判定 |
|---|---|---|---|---|---|
| gBase(mid+C) | 1.1223/1.1272 | 1.1824/1.1870 | 1.2304/1.2250 | — | 基線 |
| mid3 表(開臂 n 再 +0.12s) | 1.1239/1.1280 | 1.1783/1.1916 | 1.2244/1.2161 | ≈0 | wash → 中段預算停在 mid 表 |
| **RES_FRAC=0.45** | **1.1117/1.1173** | **1.1627/1.1797** | **1.2184/1.2138** | **−0.0103 [−0.019,−0.002] / −0.0135 [−0.023,−0.004] / −0.0116 [−0.023,−0.000]** | **promote**;n≥100 帶 −0.013/−0.013/−0.015;HPWL −0.012/−0.028/−0.005、v_rel −0.003/+0.002/−0.004;avg rt 反降 0.015–0.045s |
| mid3 + RES_FRAC=0.45 | 1.1157/— | 1.1696 | 1.2091 | ≈ RES_FRAC 單獨 | 不疊加 |

機制:refine 候選的違規修復 reserve 加大 → direct 候選 V 降 → 更多案在池內仲裁勝出(HPWL −10~−25% 的候選不再被 V 判死),tail 也受益(slice 0.5–1.1s 的 reserve 0.15–0.33 → 0.23–0.5s)。tight rung 時間減少反而縮短 runtime。分數帶:官方 100 已 **1.1117–1.1173**(自本日開工 1.135 累計 −0.02)。chain H 掃 0.35/0.6 找膝點。

### 15h. Chain H — RES_FRAC 掃描(mid 表 + C,×2)

| RES_FRAC | official r1/r2 | v3 | alpha_1 | paired vs 0.45(off / v3 / a1) |
|---|---|---|---|---|
| 0.35 | 1.1226/1.1218 | 1.1845/1.1748 | 1.2146/1.2179 | −0.0000 / +0.0015 / +0.0025 |
| **0.45** | 1.1198/1.1246 | 1.1712/1.1851 | 1.2185/1.2089 | — |
| 0.6 | 1.1259/1.1177 | 1.1841/1.1640 | 1.2317/1.2172 | −0.0004 / −0.0041 / +0.0108 |

0.35–0.6 為平台(全在噪音內),**定 0.45**。出貨候選(mid 表 + C + RES_FRAC=0.45)official 四跑 1.1117/1.1173/1.1198/1.1246(load 17–21),v3 1.163–1.185,a1 1.209–1.219。

### 15i. Chain I(出貨候選 = mid 表 + C + RES_FRAC 0.45 為基線,×2)

| 臂 | official r1/r2 | v3 | alpha_1 | paired Δ off / v3 / a1 | runtime-aware official(D=0.7/0.8/1.0,兩 rep 平均) | 判定 |
|---|---|---|---|---|---|---|
| iBase(出貨候選) | 1.1205/1.1190 | 1.1713/1.1772 | 1.2161/1.2186 | — | 0.882 / 0.853 / 0.813 | **出貨候選確認**(avg rt 0.38s,max 1.31s) |
| NREF=12 + FLOW_SLOTS=13 | 1.1348/1.1283 | 1.1544/1.1705 | 1.2260/1.2241 | +0.012 / −0.011 / +0.008 | — | 否決(official 兩 rep 皆劣) |
| tail 預算 ×1.3(n≥102,avg 0.386→0.436s,n120 1.44→1.87s) | 1.1108/1.1086 | 1.1818/1.1615 | 1.2054/1.2111 | **−0.0101 [−0.023,−0.001]** / −0.0027 / −0.0091 [−0.018,−0.002] | 0.920 / 0.887 / 0.840(**+0.03~+0.04**;超 floor 52→56 案,max rt 1.61s) | raw 有利、total 明顯不利 → 依「field 只會更快」不出貨;留作 raw/total 取捨的定價 |

Raw 1.10 目標的位置:出貨候選 official 1.112–1.125(load 17–21);再加 tail 預算可到 1.109–1.111 但 total 付 +0.03。

### 15j. 08-26 晚總結 — 出貨候選與證據

**出貨候選 env**(= run_shadow.sh 基底 + 下列覆寫;`submission/cadc1013/op_wrapper.py` 現行預設 NREF 6/OVERSAMPLE 4/無 SEAT_FIX/canonical 曲線與此不同,打包必須改成此組):

```
PARTNER_BUDGET_TABLE=$(cat artifacts/p0_newbox/budget_table_mid.txt)   # mid 表(fast 表 + n76–101 開臂)
DIRECT_CKPT=artifacts/icdc_topology/checkpoints_s2_20k/best.pt          # v2 學生(只為開臂;Direct 0 席)
FLOW_CKPT=submission/cadc1013/checkpoints/flow_matching_v1_final.pt
PARTNER_DIRECT_SEAT_FIX=1 PARTNER_NREF=9 PARTNER_FLOW_SLOTS=10
PARTNER_REFINE_RES_FRAC=0.45                                              # 新 knob(layout_refiner.py)
PARTNER_WALL_REPAIR=1                                                     # §15r(layout_refiner.py / contest_optimizer.py,default off)
PARTNER_FLOW_WARM=1                                                       # 打包安全(§15m)
PARTNER_OVERSAMPLE=1 PARTNER_KS_CAP=6 PARTNER_TAG_COMPRESS=1 PARTNER_GROUP_BRIDGE=1
PARTNER_QUOTA_FIRST 不設(default off);建議打包加 PARTNER_FLOW_WARM=1(消首案 sampler 冷啟 0.5s,constructor 內不計時)
```

| 量測 | 08-26 開工組態(§14 A:ev 表+FLOW5) | 出貨候選 | Δ |
|---|---|---|---|
| official100 | 1.152(高 load)/ 1.135(安靜) | **1.112–1.125**(4 跑,load 17–21) | −0.02~−0.03 |
| shadow v3 | 1.225 / 1.203 | **1.163–1.185** | −0.03 |
| alpha_1 | 1.235 / 1.220 | **1.209–1.219** | −0.01 |
| pseudo-hidden 121(同時段配對,舊=§9b 出貨組態) | 1.1813 | **1.1571** | −0.0242 [−0.075,+0.012];HPWL 0.127→0.097、area 0.088→0.066 |
| runtime-aware official(M=1.45,D=0.7/0.8/1.0) | 0.9196/0.8893/0.8435(fast 表) | **0.882/0.853/0.813** | −0.03~−0.04 |
| avg runtime | 0.37s | 0.38s(pseudo 0.39→0.33s) | ≈ |

- 累計機制:①中段開臂(mid 表)−0.011 ②候選違規修復 reserve 0.3→0.45 −0.010(三套 CI 皆排除 0)。兩者皆為「讓 HPWL 好 10–25% 的模型候選活過仲裁」。
- 本日否決(全部 default off / 未採):去 TAG+BRIDGE、COL_BALANCE、v4 aug 學生、partner 微調、mid2/mid3 表、QUOTA_FIRST(wash,碼保留)、FLOW_STEPS=4、FRAME_SCALE 1.00/1.01、NREF=12、RES_FRAC 0.35/0.6(平台)。
- **Raw 1.10 目標**:候選 1.112–1.125;tail 預算 ×1.3 可到 1.109–1.111 但 total +0.03~+0.04(超 floor 案 52→56),依「field 只會更快」不建議;決定權在使用者(§15i 有定價)。剩餘殘量結構:tail 違規(88/89 雙軸釘死,v_rel 0.08–0.11)、tail area gap ≈0.05(rung-0 框常數已是甜蜜點)、a1 型 pin 拓撲(column fallback 地板)。
- 程式變更(工作樹,未 commit):`partner/layout_refiner.py`(RES_FRAC knob,6 行)、`partner/contest_optimizer.py`(quota-first + toggle,bit-exact off)、`scripts/probes/quota_first_bitexact.py`、`artifacts/p0_newbox/budget_table_{mid,mid2,mid3,tail13}.txt`。回歸:14 個 partner 測試檔 169 passed / 8 skipped / 0 failed。

### 15k. Chain J — 開臂閘門重校(`PARTNER_DIRECT_SEAT_R0=0.16`,補償 reserve 0.45 後 span 縮小;×3)

| 臂 | official r1/r2/r3 | v3 | alpha_1 | 判定 |
|---|---|---|---|---|
| 候選(mid+C+RES_FRAC 0.45) | 1.1174/1.1166/1.1181 | 1.1822/1.1683/1.1683 | 1.2177/1.2143/1.2129 | 基線(三套 mean 1.1725/1.1664/1.1664) |
| SEAT_R0=0.16(閘門更嚴) | 1.1273/1.1211/1.1272 | 1.1974/1.1871/1.1914 | 1.2166/1.2164/1.2166 | **否決**:official +0.005~+0.010、v3 +0.015~+0.023 —— 關掉邊際案的臂只有損失;閘門常數維持 0.125/0.148 |

候選組態 official 七跑帶:**1.1117–1.1246**(中位 1.118)。

殘量分解(候選組態 jBase r2/r3,official 1.117,加權 excess 0.117):n<76 0.005;76–101 0.032(viol 0.015、hpwl 0.010、area 0.005);**102–120 0.080(viol 0.045–0.047、area 0.017–0.018、hpwl 0.013–0.015)** → 違規已是最大殘量(0.062),其次 tail area。Chain K 試 `VKILL=1`(違規 post-pass,carve 25% 預算)與 `PARTNER_ANYTIME_LADDER=1`。

### 15l. Chain K(候選基線 ×2)

| 臂 | official r1/r2 | v3 | alpha_1 | paired Δ off / v3 / a1 | 判定 |
|---|---|---|---|---|---|
| 候選 | 1.1119/1.1192 | 1.1716/1.1760 | 1.2188/1.2290 | — | — |
| `PARTNER_ANYTIME_LADDER=1`(r1–r4) | 1.1098/1.1182/1.1230/1.1248 | 1.1716/1.1816/1.1876/1.1870 | 1.2098/1.2146/1.2074/1.2106 | 4 reps:off +0.0040 [−0.003,+0.012] / **v3 +0.0109 [+0.001,+0.022]** / a1 −0.0142 [−0.030,−0.002];rt +0.025s | **否決**(v3 顯著退、official 不進;只在 a1 型 column-fallback 案有利) |
| `VKILL=1`(違規 post-pass,carve 25%) | 1.1247/1.1171 | 1.1847/1.1815 | 1.2136/1.2203 | +0.0053 / +0.0093 / −0.0069 | 否決(reserve 偷走 ladder 時間,v_rel 反升) |

候選組態 official 四 rep(kBase r1–r4):**1.1149 ± 0.0031**;本日全部 11 跑 1.1117–1.1246。

### 15m. Chain W — `PARTNER_FLOW_WARM=1`(打包安全性,×2)

候選 1.1219/1.1213 · 1.1668/1.1815 · 1.2142/1.2061 vs +FLOW_WARM 1.1176/1.1110 · 1.1812/1.1653 · 1.2232/1.2123:全在噪音內(分數只受首個開臂案影響)→ **打包開啟**(constructor 內暖機,不計時;消 contest 端首案 sampler 冷啟 0.5s 關臂)。

本日鏈全部收工(chains C/Mid/F/F2/G/H/I/J/K/K2/W + 兩次診斷跑 + pseudo-hidden 配對);GPU3 已釋放。

### 15n. Tail 預算曲線(raw vs runtime-aware)+ 一次 infeasible 事件 + 最後防線

| tail 預算(n≥102) | avg rt | official raw(r1/r2) | paired Δ off / v3 / a1 | runtime-aware official D=0.7/0.8/1.0 |
|---|---|---|---|---|
| ×1.0(候選) | 0.38s | 1.1241/1.1130 | — | 0.880/0.851/0.812 |
| ×1.3(§15i) | 0.42s | 1.1108/1.1086 | −0.0101 / −0.0027 / −0.0091 | 0.920/0.887/0.840 |
| ×1.6 | 0.46s | 1.1099/1.1092 | −0.0090 [−0.019,−0.001] / −0.0118 [−0.020,−0.005] / −0.0198 [−0.033,−0.008] | 0.964/0.929/0.877 |
| ×2.0 | 0.52s | 1.1090/1.1075 | −0.0103 / (v3 一案 infeasible) / −0.0221 | 1.020/0.984/0.928 |

**raw 在 ×1.3 即飽和於 ≈1.108–1.109;1.10 靠時間到不了**,而 total 每檔 +0.04。

**Infeasible 事件**(308 跑 × 100 案中唯一一次):t20_r2 v3 tid 85(n=106)block 32 目標面積 650 卻出貨 24×13=312(同 MIB 群成員 6 的尺寸;該群面積 144–650 異質 = floorset_lite data 工件)→ area_violation=1 → cost 10。其餘所有跑同案皆 feasible(MIB 違規 1–2、面積正確)→ 某後段 MIB 對齊路徑在少數候選上把共享尺寸蓋過 exact-area,且 solve() 無最後面積檢查。hidden MIB 群同質(rank-1 證據)所以曝險低,但屬硬合法性漏洞。

**修補**:`contest_optimizer.solve()` 回傳前加 `_final_area_guard`(evaluator 同形 1% 檢查;失敗則退回 post-pick 佈局,再退回 exact-area 的 column champion;`PARTNER_FINAL_AREA_GUARD=0` 可關;無違規時回傳同一物件、O(n) 純讀)。測試 `tests/test_partner_final_area_guard.py`(4)綠;三套 smoke(guard on + debug print)見下。

Guard-on smoke(sGuard_r1):official 1.1154 / v3 1.1817 / a1 1.2167,300/300 feasible,guard 零觸發(共同路徑無影響)。

### 15o. Chain L(候選基線 ×2)— tail area / 中段殘餘 fallback 的最後兩個 knob

| 臂 | official r1/r2 | v3 | alpha_1 | paired Δ off / v3 / a1 | 判定 |
|---|---|---|---|---|---|
| 候選 | 1.1187/**1.1110** | 1.1743/1.1655 | 1.2195/1.2183 | — | — |
| `PARTNER_TIGHTEN_FINE=1.0`(expand-rung 退火 floor=area_ref,STEPS 2) | 1.1139/1.1251 | 1.1811/1.1831 | 1.2178/1.2171 | +0.0046 / +0.0122 [−0.000,+0.026] / −0.0015 | 否決 |
| `PARTNER_TIGHTEN_FINE=1.01` | 1.1220/1.1164 | 1.1836/1.1647 | 1.2219/1.2171 | +0.0044 / +0.0042 / +0.0006 | 否決(wash) |
| 短 slice(<0.3s)reserve 0.6(新 knob `PARTNER_REFINE_RES_FRAC_SHORT`/`_SHORT_S`,default = RES_FRAC bit-exact) | 1.1179/1.1116 | 1.1658/1.1698 | 1.2327/1.2197 | −0.0001 / −0.0022 / +0.0073 | 否決(wash;knob 保留 default off) |

tail area gap 靠退火 floor 也回收不到(1.02 框 + 現行 0.97/0.988 退火已是 Pareto 面);1.03 臂(chain L2)補測中。

- Chain L2 `PARTNER_TIGHTEN_FINE=1.03`(×2 vs base r3/r4 1.1199/1.1154 · 1.1653/1.1727 · 1.2230/1.2214):1.1254/1.1222 · 1.1805/1.1688 · 1.2082/1.2137 → official +0.006、v3 +0.006、a1 −0.011 → **否決**。退火 floor 三檔全數否決,tail area 殘量 0.018 判為結構性(框常數/退火皆在 Pareto 面)。

### 15p. 收尾判定(08-26 22:00)

- VKILL 再審:`VKILL_RESERVE_MAX` 可縮 carve,但 §15l 的 VKILL 臂顯示 kill pass 在 direct-arm 佈局上 **v_rel 不降反升(+0.002)**、且 `kill_violations` 最少跑 0.2s → 縮 carve 也無法轉正,不再測。
- env/knob 層槓桿全部掃盡(chains C/Mid/F/F2/G/H/I/J/K/K2/W/T/S/L/L2,共 ~110 個 gate3 + 兩次診斷跑 + pseudo-hidden 配對)。候選 official 1.1149 ± 0.0031(最佳 rep 1.1110);raw 1.10 需要 tail 違規(88/89 死空間、direct 候選 ladder 後殘餘 V=4–8)的演算法工作 —— 週四窗口內屬高風險,交由使用者裁定是否投入。

### 15q. Chain O — 低預算 oversample + V-prescreen(新 knob `PARTNER_OVERSAMPLE_MIN_REM`,default 12.0 = 出貨;歷史上 <12s 剩餘時 oversample 從未啟動)

| 臂 | official | v3 | alpha_1 | 判定 |
|---|---|---|---|---|
| 候選(安靜 load 6) | 1.1159 | 1.1586 | 1.2146 | — |
| OVERSAMPLE=2(Flow 18 樣本 → prescreen 留 9;MIN_REM=0) | 1.1353 | 1.1795 | 1.2186 | **否決**(+0.019/+0.021;大批次延遲吃掉 refine 席位視窗,prescreen 估計不補償);×3 未跑 |

knob 保留(default 12.0 bit-exact)。

### 15r. `PARTNER_WALL_REPAIR=1` — 演算法層突破(deep-reasoner,08-26 深夜)— **promote,official 過 1.10**

**診斷**(`scripts/probes/wall_seat_diag.py`,對候選組態 official 佈局逐 tag 分類):n≥76 共 57 個未滿足 boundary bit → `free`(平移到牆無碰撞)28、`locked`(preplaced 釘死的 tag,牆線被 6–14 塊超出,util 91–95%)22、`swap1` 4、`blockN` 3、`interior` 0。**28 個 free 全是 cluster 成員,27/28 單獨貼牆的 dV=0**(boundary −1、grouping +1)→ 所有嚴格遞減閘門(`_edge_seat`/`_final_seat`/`_cluster_seat`/`tag_compress`)都拒絕第一步,付得起的第二步(re-weld)從未跑。tid 88/89/90/92/99 的主類是 `locked`(13/18)= 結構性,不碰。

**機制**:`layout_refiner._wall_repair`(5632)= 貼牆 + `_wall_reweld`(5777,`_cluster_seat` 的 merge 迴圈限於該群、不動持座塊)在**同一個** evaluator-form 閘門(`_violations_exact` 嚴格下降 + `_wall_hard_ok`:零 overlap、hard-shape/preplaced 位元不變、soft 面積 0.9% 內或不變差)下成交;上限 6 座位 / 25ms / 60ms 且受 deadline 夾。兩個呼叫點:ladder 修復期(6842,reserve 內)+ 最終佈局(`contest_optimizer._wall_repair_final`,991;在 `_tag_compress` 後、`_final_area_guard` 前)。off-path bit-exact(兩次 env 查詢)。離線自配對重播 1.1154→1.0967(25 案改變,0.8ms/案)。

**Gate(同鏈交錯 ×4)**:

| 套 | base | flag | paired Δ | 95% CI | v_rel | 分解 |
|---|---|---|---|---|---|---|
| official | 1.1162 ± .0023 | **1.1002 ± .0082**(1.0986/1.1031/1.0898/1.1093) | **−0.0160** | [−0.0256, −0.0079] | .0298→.0220 | boundary 加權 1.176→0.735(−37%)、grouping .504→.435、HPWL/area 持平 |
| v3 | 1.1724 | **1.1493** | **−0.0230** | [−0.0418, −0.0046] | .0446→.0387 | HPWL −0.012 |
| alpha_1 | 1.2144 | **1.2046** | −0.0098 | [−0.0232, −0.0005] | .0336→.0314 | — |

1200/1200 feasible;avg rt +1.7%(0.375→0.381s),p99 1.24→1.31;runtime-aware(M=1.45)三個 D 全部改善(0.881/0.852/0.813 → 0.872/0.842/0.803)。歸因(×2):final-site-only −0.013/−0.020/−0.002;兩站 vs 只 final −0.004/−0.008/−0.004 → 兩站都留。修後 `free` 類 28→1;殘餘 = `locked` 22(packing-density 問題)。測試 `tests/test_partner_wall_repair.py`(8)+ 全 partner 回歸 1062 passed / 0 failed。**判定:promote;出貨 env 加 `PARTNER_WALL_REPAIR=1`。**

### 15s. 最終出貨 env 確認(mid 表 + C + RES_FRAC 0.45 + WALL_REPAIR + FLOW_WARM,×2)

zFinal r1/r2:**official 1.0973 / 1.0908**、v3 1.1550 / 1.1445、a1 1.2005 / 1.1925;600/600 feasible。與 WALL_REPAIR gate 的 4 reps 合計 official 六跑 1.0898–1.1093,**均值 1.0982 ≤ 1.10 目標**。本日累計:1.135–1.152 → 1.098(−0.04~−0.05),v3 1.225 → 1.15,a1 1.235 → 1.20。

## 16. Beta hidden 落差真因(08-27 凌晨,官方逐案 JSON)— **部署失效,不是資料分佈**

`docs/official/beta_test/cadc1013.tar.gz` = 官方對我們 beta 包的逐案結果(`beta_evaluation_results.json`)+ `eval_op_wrapper.log`。

| 加權 excess | n<76 | 76–101 | 102–120 |
|---|---|---|---|
| **hidden beta(1.3141)** | 0.010 | 0.062 | **0.242 = hpwl 0.115 + area 0.027 + viol 0.083** |
| local v3 @beta 組態(1.2914) | 0.010 | 0.092 | 0.189(hpwl 0.041、viol 0.104) |
| local alpha_1 @beta 組態(1.2307) | 0.008 | 0.052 | 0.170(hpwl 0.077) |
| local official @final(1.0908) | 0.005 | 0.024 | 0.062 |

- hidden tail **每一案** hpwl gap 0.14–0.69 = column fallback 簽名(本機 tail 0.02–0.08)。
- runtime 對照 canonical 預算:n≤98 貼預算(臂閘門關);**n=99(第一個預算 >0.3s、臂開)起 runtime 1.27–1.64s,超出 +1.05s(n=99)→ +0.36s(n=120)** → Flow sampler 在 contest 機約 1s(本機暖機 0.08–0.13s),refine 席位視窗負 → 候選全 None → tail 全退 column;每案白付 ~1s(beta total/raw 0.745 未貼 floor 的原因)。
- log:`[polish] scipy unavailable; pass disabled`(coord_polish 整場關閉);`submission/cadc1013/requirements.txt` 為 **0 bytes** → contest venv 用他們的 base 環境(無 scipy,torch 疑為 CPU 版)。
- **本機重現**:`CUDA_VISIBLE_DEVICES=""`(CPU torch)跑 final env official 100 → **1.2981、hpwl 0.33、avg rt 0.81s、max 2.06s**(GPU 1.09 / 0.05 / 0.37s)= hidden 簽名。

**結論**:alpha_1 的「column 地板」確實是 beta 發生的事,但原因是部署(模型通道沒跑起來),不是 hidden 題目像 alpha_1。修法:
1. `submission/cadc1013/requirements.txt` 已寫入 `torch>=2.5.0 numpy>=1.26.4 numba>=0.61.0 scipy>=1.13.0 tqdm>=4.66.4`(partner 路徑第三方 import 清單:torch/numpy/scipy/numba;evaluator 用 tqdm)。
2. `_warm_flow_sampler`(PARTNER_FLOW_WARM=1)新增 **`[selfcheck]`** 一行(cuda_available、device、暖機延遲)+ **adaptive seat gate**:第二次暖機量測的 warm latency 若 > 1.25× `PARTNER_DIRECT_SEAT_TS` 假設值,自動抬高 `PARTNER_DIRECT_SEAT_TS`(`PARTNER_SEAT_TS_ADAPT=0` 關)→ sampler 太慢時不開臂,至少不再每案白付 1s(品質仍需 GPU 到位)。驗證跑:CPU(adapt)+ GPU(final env)見下。

驗證(final env,official 100):

| 環境 | selfcheck | noRT | avg rt | max rt |
|---|---|---|---|---|
| CPU torch,無 adapt(cpuFinal) | — | 1.2981 | 0.807 | 2.06 |
| **CPU torch + adapt**(cpuAdapt) | cuda=False, warm 0.996s → seat_ts 0.996 | 1.2679 | **0.346** | 1.21 |
| **GPU**(gpuAdapt) | cuda=True, warm 0.068s → kept 0.148 | **1.0901** | 0.385 | 1.42 |

CPU 暖機 0.996s = contest 機每案多付的 ~1s;adapt 後 runtime 回到預算(column-only 品質 1.27 仍差,真正的修法是 GPU torch 到位),GPU 路徑 bit-for-bit 不變(kept)。決賽包若 GPU 正常,hidden raw 期望回到 v3 水準(≈1.15–1.18);若又落到 CPU,至少 rt_adj 貼 floor。

### 16a. 打包演練第一輪(08-27 04:30)發現

- 乾淨 venv 只裝我們的 requirements 後,**官方 evaluator 自己 import 失敗**(`FloorSet/lite_dataset_test.py` 需要 `requests`;其 requirements 還有 matplotlib/shapely/tqdm)。Case B 規則是「venv 只用我們的檔案建」且 evaluator 也在裡面跑 → requirements 必須 = 我們的依賴 ∪ 官方 `FloorSet/iccad2026contest/requirements.txt`(torch/numpy/shapely/matplotlib/tqdm/requests)+ 指南列出的 scipy/numba/threadpoolctl。已更新 `partner/shipping/requirements.txt`。
- 未釘版的 `torch` 在 PyPI 解成 **2.13.0+cu130**(本機 driver 580/CUDA 13.0 可用);contest A100 的 driver 未知,cu130 wheel 需 driver ≥580,否則 `cuda_available=False` → 重演 beta。改釘 **`torch==2.6.0`**(預設 wheel = cu124,driver ≥525 即可)。
- **本機主 venv 也沒有 scipy**:08-21 遷移後所有 gate 跑都印了 `[polish] scipy unavailable; pass disabled`(每跑 26 案)→ 出貨候選的所有數字都是 coord_polish **OFF** 量的;打包 venv 有 scipy 會變 ON。已 `uv add scipy`,chain P 同鏈量 `PARTNER_COORD_POLISH=1` vs `0`(×2)決定包內設定。

### 16b. Chain P — coord_polish 真 ON/OFF(scipy 裝入主 venv 後;OFF = 變數 unset,`=0` 仍為 ON)

| 臂 | official r1/r2 | v3 | alpha_1 |
|---|---|---|---|
| polish OFF(= 08-21 以來所有 gate 的實際狀態) | 1.1060/1.1060 | 1.1575/1.1582 | 1.2286/1.2121 |
| **polish ON**(打包 venv 的實際狀態) | **1.0886/1.0957** | **1.1401/1.1364** | **1.1946/1.1985** |

→ 打包環境會比本機 gate 再好一截;`PARTNER_COORD_POLISH=1` 維持,requirements 帶 scipy。本機主 venv 已 `uv add scipy`(pyproject/uv.lock 變更),之後的 gate 都是 polish ON 口徑。

### 16c. 打包演練第二輪(08-27 06:40)— 乾淨 Py3.13 venv 只用新 requirements → 官方 evaluator 100 案

- venv:torch **2.6.0+cu124**、`cuda_available=True`、scipy 1.18.1、numba 0.67.0、numpy 2.2.6、requests/shapely/matplotlib 齊。
- log:`[selfcheck] cuda_available=True device=cuda flow_warm_latency=0.069s seat_ts=kept 0.148`;`loaded direct model step 18000`(v2 學生)、`loaded flow model step 1000000`;`scipy unavailable` 0 行(polish ON)。
- 官方 evaluator:**1.0935,100/100 feasible**;逐案 runtime 與本機同組態(pOn_r1)中位差 −0.001s,tail hpwl 0.00–0.16 → 模型通道正常。
- **抓到一個打包 bug**:worker 以 `contest_optimizer.py` 的 import 閉包組包,漏掉 `op_wrapper.py` 自己 import 的 `tests/synth_instances.py`(JIT 暖機用)→ 暖機靜默失敗,**第一案(n=21)付 10.5–11.2s numba 編譯**(beta 舊包有帶,hidden 第一案 0.08s)。補入後冷 cache 第一案 0.052s。包內 __pycache__ 清掉後重打:`cadc1013.tar.gz` 31 條目。
- 包內無絕對路徑;tar 全部在 `cadc1013/` 下。

### 16d. Chain Q — polish 時間上限掃描 + 出貨決定

| 臂(polish) | official r1/r2 | v3 | a1 | paired vs 300ms(off/v3/a1) | avg rt | runtime-aware official M=1.45 D=0.7/0.8/1.0 |
|---|---|---|---|---|---|---|
| OFF(unset) | 1.1060/1.1060 | 1.1575/1.1582 | 1.2286/1.2121 | +0.0138 / +0.0196 / +0.0238 | 0.386 | **0.881 / 0.852 / 0.812** |
| ON,300ms(default) | 1.0908/1.0976 | 1.1389/1.1395 | 1.2100/1.2032 | — | 0.455 | 0.935 / 0.901 / 0.849 |
| ON,100ms | 1.1089/1.0978 | 1.1425/1.1545 | 1.2178/1.2203 | +0.0092 / +0.0093 / +0.0125 | 0.422 | 0.915 / 0.881 / 0.832 |
| ON,40ms | 1.1035/1.1126 | 1.1542/1.1567 | 1.2223/1.2246 | +0.0139 / +0.0162 / +0.0169 | 0.420 | 0.914 / 0.881 / 0.835 |

- 上限版把 raw 增益幾乎全吐回、runtime 卻只省一半 → 只有「全開」或「全關」兩個選項。
- 全開 raw −0.014,但 total 在每個情境都輸:M=1.0/D=1.0(最樂觀)+0.010,M=1.45/D=0.7 +0.05。polish 是 post-deadline pass(tail 26 案每案 +0.25s),與 tail 預算 ×1.3 同類的 raw/total 取捨(每秒價值相近)。
- **出貨包定為 polish OFF**(`PARTNER_COORD_POLISH` 不設;total 口徑在所有 M/D 情境勝);要換 raw 只需在 op_wrapper 加一行 `PARTNER_COORD_POLISH=1`。本機 gate 口徑自 chain P 起改為 polish 可用但 OFF = 與出貨一致。

### 16e. 最終包演練(polish OFF,含 synth_instances;`submission/cadc1013_0827_final.tar.gz`,md5 `ee7628ff225317e1b9f02f35ad854f54`,1.20 GB,34 條目)

全新解壓 + 乾淨 Py3.13 venv(torch 2.6.0+cu124)+ 官方 evaluator:`[selfcheck] cuda_available=True … flow_warm_latency=0.069s`;direct step 18000 / flow step 1000000 載入;**noRT 1.1147,100/100 feasible,avg rt 0.406s,p90 0.91,max 1.48,第一案 0.057s**(JIT 已在載入期)。與本機 polish-OFF 口徑(1.09–1.106)同帶。

週五上傳前只剩:①若要 raw 優先,op_wrapper 加 `"PARTNER_COORD_POLISH": "1"` 重打;②branch 推上 GitHub(需憑證)。

## 17. 目標 official 1.08 / v3 1.10(08-27 使用者裁定;悲觀評估 M=1.45、D=0.6;A100+Icelake)

### 17a. `PARTNER_PIN_FRAME`(deep-reasoner)— 機制成立、分數未過,**default off**

- 診斷(`PARTNER_PINFRAME_DEBUG=1`,`[pf]` 逐候選):`locked` 類的超出在 ladder rung 內產生,之後任何 post-pass 都不會移除;`lock_*` 在出貨路徑只當「縮框下限」(`_tighten`/`_try_squeeze`/`compact_to_locks`),從不當上限;min 側 lock 在主路徑完全沒用到(rung 0 的 W 用預測 bbox 的 xmin 算)。tid 75/76/83 無 direct 候選(column 勝),機制碰不到。
- 機制:`_Refiner._pin_frame_to_locks()`(layout_refiner.py:2065)把鎖側夾到牆線、把面積補到另一自由邊;三個呼叫點(rung 0、rung-0 salvage、tight expand rungs)。off-path 6/6 bit-exact;10 測試綠(共 41)。
- Gate(×2 交錯,polish off):official −0.0022 [−0.032,+0.028]、**v3 +0.0121**、a1 +0.0070;v_rel 降(−0.0008/−0.0018)但 HPWL 升(v3 +0.0245);runtime 反降(avg 0.394→0.377,max 1.47→1.27);M=1.45 D=0.6 runtime-aware 0.933→0.919。子模式 `min`(只修 rung-0 角)+0.006/+0.002/+0.008;`RETRY` 更差(+0.008/+0.022/+0.020)。
- 逐案:tid 86 1.316→1.054、83 1.270→1.099 大勝;tid 99 1.187→1.318、69 1.106→1.281 大敗(無違規的案被硬夾框)。locked 26→22(official)、32→20(v3)。
- **結論**:locked 類不是免費的 —— 夾框換來 HPWL;要拿 86/83 的贏而不吃 99/69 的輸,得走 portfolio(pinned 候選與 unpinned 同池、逐案仲裁),不是 ladder 全域改。

### 17b. `PARTNER_PIN_FRAME_SLOTS=k`(portfolio 席位;deep-reasoner round 2)— 機制成立、彙總未過,default 0

- 管線:pin 旗標走 payload(fork pool 繼承 env,不能用 env 區分同案兩個 worker;仿 `opt._tag_anchor`),`_worker_refine` 讀第 12 個 payload 元素;`_has_tag_locks(opt1)` 只在有 preplaced-tag 的案啟用(official 76/v3 73/a1 76 案);`_pin_slot_specs` 從 specs 尾端(tag-anchored extras)轉換 k 個 → 不多花 worker/預測/runtime。17 新測試 + 24/117 綠;off-path 8/8 bit-exact。
- Gate(4 條鏈、22 個 GATE3、66 跑全 100/100):**SLOTS=2 合併 4+4 reps:official −0.0115 [−0.0282,+0.0003]、v3 +0.0020、a1 +0.0056**;runtime-aware(M=1.45)D=0.6/0.7/0.8 全部 −0.008;runtime ±1%。SLOTS=1/3、+PSEL_EXACT_V、+PSEL_FIX 皆不成 composite(EXACT_V 首次在 a1 為正 −0.019,但 v3 不救)。
- 逐案:global 版的 tid 99(+0.13)/69(+0.175)損失消失;tid 86 1.297±0.048 → **1.069±0.042、bnd 0(16/16 跑)**,單案 = official 增益的 −0.0064(幾乎全部);v3 沒有這種 locked 大案,只付「拿走一個 anchored 候選」的 HPWL 成本。
- **判定:default 0**;後續兩個便宜方向:①席位改從 plain draw 拿(不拿 anchored extra)②以 lock box utilization 門檻縮小啟用集(76/100 → 少數 binding 案)。

### 17c. alpha_1 是不是 hidden 的樣子?(08-27,使用者朋友回報「alpha_1 ≈ 他們的 beta」)

alpha_1 manifest:GT 幾何/pins/B2B/constraints/每 pin 的 P2B 度數皆保留,只把每個 pin 連到哪個 block 用 P∝(1−z)(z=pin→GT block 中心的正規化曼哈頓距離)重抽。量 P2B 距離結構(連線 block 在該 pin 距離排序的百分位;越小越近):

| 套 | mean_rank_pct | 連到最近 10% 的比例 |
|---|---|---|
| official(LiteTensorDataTest) | **0.038** | **91.6%** |
| v3(官方生成器新生成) | **0.039** | 90.9% |
| alpha_1 | 0.366 | 18.1% |

→ 官方生成器幾乎把 pin 接到最近的 block,v3(同生成器新 instance)完全一致;alpha_1 是我們造的 10× 隨機化**合成偏移**,不是生成器會產出的分佈。hidden 用同一生成器 → pin 結構應像 official/v3。中段(n=76–98,beta 預算下三套皆 column-only)hpwl gap:hidden 0.321、official@canonical 0.408、alpha_1 0.299、v3 0.478 —— 跨套 hpwl_gap 受 golden 基準變動干擾,不能單獨判別;P2B 統計是主證據。朋友的「像」需三個數(local official / local alpha_1 / beta)才能分辨是 solver 對 pin 不敏感(official≈alpha_1≈beta)還是真有偏移。**保險**(deadline 延至 8/31):把模型條件中的 p2b 中性化/降權,三套 gate,official/v3 不掉且 a1 大進才上。

### 17d. Round 3 — 席位來源 + lock-box utilization 門檻(×3 交錯,polish off)

| 臂(SLOTS=2) | official r1/r2/r3 | v3 | a1 | paired Δ off / v3 / a1(3 reps) |
|---|---|---|---|---|
| base | 1.0985/1.0967/1.1026 | 1.1430/1.1460/1.1435 | 1.2128/1.2072/1.2119 | — |
| plain 來源 | 1.1003/1.1037/1.0986 | 1.1602/1.1497/1.1500 | 1.2323/1.2100/1.1956 | +0.0016 / **+0.0091 [+0.002,+0.015]** / +0.0020 → 否決 |
| plain + MAX_UTIL 0.75 | 1.1150/1.0979/1.0933 | 1.1558/1.1474/1.1465 | 1.2183/1.2033/1.1979 | +0.0028 / +0.0057 / −0.0042 → wash |
| **anchored + MAX_UTIL 0.75**(`PARTNER_PIN_FRAME_SLOTS=2 PARTNER_PIN_FRAME_SLOT_MAX_UTIL=0.75`) | **1.0923/1.0899/1.0889** | 1.1450/1.1400/1.1520 | **1.1903/1.1976/1.1937** | **−0.0089 [−0.022,+0.003] / +0.0015 [−0.004,+0.007] / −0.0168 [−0.035,−0.003]** |

anchu:三套 mean −0.008,official/a1 3/3 同向,v3 wash;runtime 持平(0.378→0.383);M=1.45 runtime-aware D=0.6/0.7/0.8 base 0.904/0.867/0.838 → 0.899/0.861/0.832。判準「official 與 v3 皆進步」在 v3 上是 wash 而非進步;依使用者較早判準(v3+a1 mean −0.008、official 不退)可促轉。**暫列出貨候選 B**,待 deep-reasoner 報告確認 live-case 數與機制後決定是否入包。

Deep-reasoner round-3 報告補充:①`MIN_UTIL`(只在緊的 lock box 開席)剛好殺掉會贏的案(tid 86 util 0.696);有效的是 **`MAX_UTIL`**(lock box 還有鬆弛才開席):official/v3/a1 live 案 50/40/15(u≤0.75),pipeline 內 official 實際 28 案開席。門檻是寬平台(u=0.70–0.90 皆 −0.006~−0.008),leave-one-suite-out 皆勝 ungated。②anchu 增益是 **HPWL 驅動**(−0.006/−0.008/−0.018)、v_rel 持平 —— 門檻把「用 wirelength 換 boundary」的案擋掉了。③runtime avg +1.3%、max +4%(re-ladder 既有 worker,在漂移內)。④**地雷**:`legalize_rectangles` 把 `_parallel_solve` 的任何例外吞掉(`except Exception: pass`)靜默退回 sequential → 一個 debug 行的 NameError 讓 official 變 1.328 而無任何錯誤;已在該 except 加 stderr 警告(見下)。⑤未關閉的 caveat:anchu 每 rep 都排第 4 位(位置混淆);u=0.75 是 in-sample 選的;a1 的 util 分佈與 official 不同(開席比例 50/40/15)。**下一步:反序鏈(anchu 先、base 後)×2 才能促轉。**

### 17e. p2b 保險 gate(`PARTNER_COND_P2B=off`:模型條件不看 pin,legalizer/refiner 仍優化真 p2b;×2 交錯)

| | official | v3 | alpha_1 |
|---|---|---|---|
| base | 1.0918/1.0903 | 1.1482/1.1514 | 1.1994/1.2012 |
| p2b off | 1.1459/1.1527 | 1.2012/1.1996 | **1.1307/1.1419** |
| paired Δ | **+0.054** | **+0.051** | **−0.064** |

- 模型的 pin 條件在真實生成器分佈(pin 接最近 block)上值 0.05;在 alpha_1(pin 隨機化)上反而害 0.06–0.07 → alpha_1 分數差的真因就是「模型相信會說謊的 pin」。
- 全域關閉不可出貨;若 hidden 真有 pin 偏移(需朋友的 official/alpha_1/beta 三數證實),可做 portfolio 席位版(少數 worker 用 p2b-off 候選)當對沖,估 official +0.01 換 a1 −0.04。未證實前不做。
- 悲觀模擬(預算 ÷1.45,×2):official 1.1526/1.1560、v3 1.2600/1.2489、a1 1.2285/1.2255 —— 上界(門檻也被縮,中段過罰);門檻同比例縮的修正版 emu145b 排隊中。

### 17f. anchu 反序確認 → **不促轉**

反序(anchu 先、base 後)×2:official +0.0024 [−0.004,+0.009]、v3 +0.0006、a1 −0.0098。**合併 5 reps**(3 正序 + 2 反序):official −0.0044 [−0.0132,+0.0043](1.0957±0.0054 → 1.0913±0.0029)、v3 +0.0011、a1 −0.0140 [−0.031,−0.000];runtime-aware M=1.45 三個 D 完全相同(0.899/0.861/0.832)。正序 3 reps 的 official −0.009 是位置混淆(anchu 固定排第 4)。只有 a1(合成 pin 偏移套)真有增益 → 依「hidden 像 official/v3」的判定,**default 0,不入包**;若朋友的三數證實 hidden 有 pin 偏移,可與 p2b-off 席位一起重新考慮。

修正版悲觀模擬 emu145b(預算 ÷1.45 + 門檻同比例縮):official 1.1374 / v3 1.1661 / a1 1.2279 → contest 機 raw 期望區間 official 1.11–1.14、v3 1.15–1.17(M 1.2–1.45)。

### 17g. `PARTNER_EARLY_EXIT=1`(SA 收斂提前結束;×2,EE 先 base 後,load 30–53 高噪音)

| | official | v3 | a1 | avg rt(off/v3/a1) | max rt |
|---|---|---|---|---|---|
| base | 1.1055/1.1097 | 1.1530/1.1898 | 1.2306/1.2241 | 0.402/0.419/0.357 | 1.48 |
| EE | 1.0960/1.1010 | 1.1481/1.1552 | 1.2126/1.2186 | **0.371/0.373/0.349** | 1.36 |
| paired Δ | −0.0091 [−0.023,+0.001] | −0.0198 [−0.040,−0.006] | −0.0117 | **−8%** | |

runtime-aware M=1.45 D=0.6 0.938→0.900。raw 的「更好」可疑(EE 固定先跑、load 從 30 升到 53,base 吃到尖峰);runtime 減少為真(提前退場)。→ 反序五套鏈(base 先、EE 後,含 v5/v6)×2 確認。

### 17h. 新 shadow 套 v5 / v6 基線(候選組態,polish off;load 53–64 高噪音)

v5(quantile 0.20、零 golden-MIB、public-shift mean 0.84):**1.1287 / 1.1331**(hpwl 0.06–0.07、area 0.07、v 0.025–0.030);v6(quantile 0.50、零 MIB):**1.1429 / 1.1717**(rep 間差 0.03,load 尖峰)。序列:public 1.10 < v5 1.13 < v6 1.14–1.17 ≈ v3 1.145。判準自此 = public + v3 + v5/v6(五套 runner `run_gate5.sh`),alpha_1 順便。

反序五套鏈 ×2(base 先、EE 後)+ 正序 ×2 合併:

| 套 | base | EE | paired Δ | 95% CI |
|---|---|---|---|---|
| official(4 reps) | 1.1113 ± 0.0055 | **1.1039 ± 0.0076** | **−0.0073** | [−0.0152, −0.0015] |
| v3(4) | 1.1645 | **1.1543** | **−0.0102** | [−0.0203, −0.0013] |
| v5(2) | 1.1474 | **1.1256** | −0.0218 | [−0.0431, +0.0003] |
| v6(2) | 1.1388 | **1.1302** | −0.0086 | [−0.0194, −0.0009] |
| a1(4) | 1.2302 | 1.2175 | −0.0127 | [−0.0260, −0.0020] |

avg runtime 0.407 → 0.383(−6%);runtime-aware M=1.45 D=0.6/0.7/0.8:0.939/0.899/0.868 → **0.912/0.873/0.844**。兩種臂順序皆同向,五套全進步,runtime 降 → **promote `PARTNER_EARLY_EXIT=1`**(出貨 env 加入;raw 增益機制推測 = 提早釋放收斂的 column restart,refine worker 少搶 CPU;contest 機獨佔時 raw 效果可能較小,runtime 效果仍在)。

### 17i. `PARTNER_REFINE_SECURE_FALLBACK=1`(deep-reasoner)— **promote**

- 真因(儀器化全跑):`refine_prediction` 先扣 `res=0.45·slice`,rung 0(1.02 框,95% util)在失敗帶 **522/522 次全失敗**(殘餘 overlap 1–87 塊);expand 迴圈第一個 +2% rung **無 deadline**(`_rdl=None`,:6931)吃光剩餘 → 0.05/0.08/0.12/0.18/0.28 rung 在任何出貨預算下**從未執行** → `legal is None` → 整個 reserve 丟掉。**522 次 ladder 中 120 次(23%)如此,分佈在 34 個 n**,不只 75/76/83。
- 機制(layout_refiner.py:411 `secure_fallback_on`、:6386 `_secure_fallback`、hook :7185):ladder 全失敗且時間尚餘時,跑一個**有 deadline** 的寬框 rung(預設 expand 0.12,0.28,跳過已被駁回的),鏡像出貨的 pin-less rung(expand → anchor → legalize_soft → tag recovery → `_tighten` → clusters),只用失敗 ladder 沒花掉的 reserve,絕不超過 `t_hard`。off-path 一次 env 查詢;30 測試綠(合 186 passed)。
- Gate(五套 ×2,臂順序對調,polish off):

| 套 | paired Δ | 95% CI | 分解 |
|---|---|---|---|
| official | **−0.0097** | [−0.0179, −0.0020] | hpwl 0.078→0.059、area −0.004、v 持平 |
| v3 | **−0.0077** | [−0.0161, −0.0001] | hpwl −0.013 |
| v5 | **−0.0135** | [−0.0271, −0.0018] | hpwl −0.016、area −0.013 |
| v6 | −0.0046 | [−0.0132, +0.0017] | — |
| a1 | −0.0167 | [−0.0342, −0.0024] | hpwl −0.024 |

runtime avg 4/5 套降、p90 5/5 降(official 0.410→0.396 / 0.969→0.882),max 三套微升(單案尾);runtime-aware M=1.45 三個 D 皆 **−0.018**;20/20 跑 100/100 feasible。tid 75/76/83 各 −0.050/−0.008/−0.059,但主要增益來自其他 ~34 個 n 的 fallback 候選勝出。品質門檻子模式(`MAX_BBR`/`MAX_VREL`)被支配,default off。
- **附帶發現(下一個槓桿)**:出貨 ladder 的 0.05–0.28 rung 在出貨預算下是死碼,+2% rung 無界 → ladder 值得獨立重新配預算(bound 0.02 rung、讓寬 rung 可達)。

### 17j. 合併確認 EARLY_EXIT + SECURE_FALLBACK(五套 ×2,順序對調)

| 套 | EE only | EE + SF | paired Δ | 95% CI |
|---|---|---|---|---|
| official | 1.1097/1.1099 | **1.0934/1.1056** | **−0.0103** | [−0.0197, −0.0011] |
| v3 | 1.1513/1.1451 | 1.1435/1.1607 | +0.0039 | [−0.0043, +0.0127] |
| v5 | 1.1124/1.1015 | 1.1076/1.1051 | −0.0006 | [−0.0097, +0.0077] |
| v6 | 1.1384/1.1153 | **1.1209/1.1130** | **−0.0099** | [−0.0238, −0.0008] |
| a1 | 1.2230/1.2051 | 1.2179/1.2221 | +0.0059 | [−0.0049, +0.0175] |

runtime 持平(0.381→0.382);runtime-aware M=1.45 D=0.6 0.918→0.906。疊在 EE 上 SF 仍正(official/v6),v3/v5 wash(EE 已釋放部分 CPU 給 refine,增益重疊),無一套變差 → **兩者皆入出貨 env**。包重打(op_wrapper 含 EARLY_EXIT + SECURE_FALLBACK),乾淨 venv 演練中。

### 17k. 打包演練 3/4 與一個自傷 bug

- 演練 3(md5 11d7969a):包內模組是 04:27 的舊版(缺 SECURE_FALLBACK 等後續碼)→ **打包必須從工作樹重組**;已寫 `scripts/pack_cadc1013.sh`(import 閉包 + op_src + `partner/shipping/op_wrapper.py` 模板 + requirements + ckpt + synth_instances,內容與手工包逐檔 md5 相同)。
- 演練 4(重組後,md5 eb5b1937):`[selfcheck] … cpu_ratio=1.91 seat_r0=adapted->0.2386` → **CPU 自校準在共用機負載下誤判 1.91×,把 R0 門檻抬高、關掉大半模型臂,official 1.1315**。而 ÷1.45 模擬已證明慢 CPU 上開臂仍划算(1.137 vs 關臂 1.155)→ R0 自校準改為 **opt-in(default off)**,只印 cpu_ratio 供診斷;TS 自校準(sampler 真的 1s 時關臂)維持。

- 演練 5(R0 自校準 off,`scripts/pack_cadc1013.sh` 組包,md5 **163b885410c02ec021b6c3699d8cd62d**):`[selfcheck] cuda_available=True … seat_ts=kept 0.148 cpu_ratio=0.62 seat_r0=kept`(cpu_ratio 在同一台機器上 0.62↔1.91 漂移,證明它不能當閘門)、polish off、**noRT 1.1042,100/100,avg rt 0.370,max 1.39,第一案 0.051s**(load 29)。→ `submission/cadc1013_0827_final.tar.gz` 更新為此包(含 EARLY_EXIT + SECURE_FALLBACK)。

### 17l. `PARTNER_LADDER_REBUDGET`(deep-reasoner)— 機制成立、分數部分過,**hold**

- 機制(layout_refiner.py:480 `ladder_rebudget_on`、rung 列表改寫 :7157、預算計畫 :7180、逐 rung deadline :7248):+2% rung 綁 `R1_FRAC·span`,後續 rung(預設 0.05,0.12,0.28)平分剩餘、最後一級拿餘量;rung 0 不動;被時間截斷的 rung 不向 SECURE_FALLBACK 報「已駁回」。43 測試綠、off-path bit-exact。附加 `PARTNER_LADDER_SECURE_MIN`(default 0 = inert):保留給 escape rung 的最小 span。
- Census(official 522 次 ladder):base 的 45% `rung_cap` 其實直接跳到 0.28 escape(0.28 跑 168 次、成交 165);REBUDGET 把 ~126 次 loose-frame 成交換成 0.05/0.12 的 tight-frame 成交(增益來源),代價是 escape rung 只剩殘餘時間、成交率 98%→42%,ladder 失敗 124→149(由 fallback 接手)。
- Gate(五套、4 輪交錯、順序對調;`R1_FRAC=0.35` 勝 0.5):4 輪 official −0.016 但**全靠一輪 base 吃 load 尖峰**(avg_rt 0.484);去掉那輪:official −0.000、v3 −0.007、v5 +0.001、**v6 −0.012 [−0.023,−0.001]**、a1 −0.012 [−0.028,−0.001];runtime 不升,runtime-aware 中性(+0.0015)。
- → public/v5 持平、v3/v6/a1 進步 = 部分達標。缺的一塊 = escape rung 視窗:chain L3 測 `REBUDGET=1 R1_FRAC=0.35 SECURE_MIN=0.30` vs base(×2 對調)。

### 17m. Chain L3 — `REBUDGET=1 R1_FRAC=0.35 SECURE_MIN=0.30` vs base(五套 ×2 對調,load 27–47)

| 套 | base r1/r2 | RS r1/r2 | paired Δ | 95% CI |
|---|---|---|---|---|
| official | 1.1003/1.0914 | 1.1217/1.0903 | **+0.0102** | [−0.0046, +0.0259] |
| v3 | 1.1554/1.1473 | 1.1503/1.1453 | −0.0035 | [−0.0154, +0.0072] |
| v5 | 1.1152/1.1098 | 1.1001/1.1083 | −0.0083 | [−0.0183, +0.0035] |
| v6 | 1.1353/1.1417 | 1.1226/1.1185 | **−0.0180** | [−0.0301, −0.0091] |
| a1 | 1.2250/1.2202 | 1.2290/1.2176 | +0.0007 | — |

runtime avg 0.394→0.383;runtime-aware official 混合(r1 +0.018、r2 −0.013)。**public 不進反退(rep 1 +0.021 主導,rep 2 −0.001)、v6 大進、v3/v5 小進** → 未達「public 不退」判準,hold;再排 2 rep(chain L4)解 public 的歧義。

### 17n. Chain L4(再 2 rep,順序對調)→ 合併 4 reps 判定:**hold(不入包)**

| 套 | base(4) | RS(4) | paired Δ | 95% CI |
|---|---|---|---|---|
| official | 1.0989 ± 0.0052 | 1.1005 ± **0.0143** | +0.0017 | [−0.0094, +0.0115] |
| v3 | 1.1531 | 1.1465 | −0.0066 | [−0.0185, +0.0038] |
| v5 | 1.1121 | 1.1097 | −0.0024 | [−0.0118, +0.0087] |
| v6 | 1.1343 | **1.1250** | **−0.0094** | [−0.0192, −0.0022] |
| a1 | 1.2217 | 1.2185 | −0.0032 | — |

runtime 持平(0.384→0.382);runtime-aware official 中性(+0.002)。public wash 且 RS 臂 rep 間變異是 base 的 3 倍(1.090–1.122)→ 對單次 hidden 評分是風險;只有 v6 顯著。依「public 不退且 v3/v5/v6 進步」判準不足,**保留 default off**。出貨包維持 md5 163b8854(EE + SF)。
