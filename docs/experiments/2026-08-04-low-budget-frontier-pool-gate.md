# 2026-08-04 低預算 Q-time 前緣 + pool 閘門斷崖(goal: noRT 1.000 / avg 0.2s)

## 背景

使用者設定終局目標:**total_score_no_runtime → 1.000、平均 runtime → 0.2s/case**。
基準(0730 定版):noRT 1.128–1.135 @ avg 2.0s(0729 定案 env:`PARTNER_DIRECT_SOLVER=dpmpp
PARTNER_DDIM_STEPS=10 PARTNER_REFINE_STALL_STOP=1 PARTNER_BUDGET_MAX=3.5 PARTNER_DIRECT_MIN=2.0`
+ flow st8 antithetic)。品質缺口 −0.13、時間 ÷10,而已知 cliff#2@3s。

## 實驗 1:低預算前緣掃描(chain `scratchpad/lbf_frontier_chain.sh`,7×full-100,乾淨機器)

| arm | noRT | proj | avg_rt | tailQ | feas |
|---|---|---|---|---|---|
| ctrl35 rep1 | 1.1318 | 0.8755 | 1.99s | 1.1210 | 100 |
| MAX=2.0 | 1.3524 | 0.9547 | 1.42s | 1.3538 | 100 |
| MAX=1.0 | 1.3855 | 0.9699 | 0.90s | 1.3876 | 100 |
| MAX=0.5(MIN=0.3) | 1.4239 | 0.9967 | 0.46s | 1.4269 | 100 |
| MAX=0.5+DM0.05 | 1.4192 | 0.9935 | 0.46s | 1.4200 | 100 |
| MAX=0.25(MIN=0.15) | 1.4890 | 1.0423 | 0.25s | 1.4969 | 100 |
| ctrl35 rep2 | 1.1284 | 0.8739 | 2.01s | 1.1170 | 100 |

發現:
1. **0.2s/case 無物理障礙**:實際牆鐘緊貼預算(excess P50≈0.00、P90≈0.01),100/100
   feasible 全程不破。runtime 軸純粹是品質代價問題。
2. **彈性劇烈非線性**:24→3.5s 只付 +0.03(0723 已證)、3.5→2.0s 付 **+0.222**、
   2.0→0.25s 只再付 +0.14。
3. 逐案分解(3.5 vs 2.0):**81% 是 HPWL**(+0.180/+0.222),均勻散佈全 tail
   (每個 n≥100 案 dHPWL +0.2~0.66);area +0.023、V_rel +0.017。

## 實驗 2:斷崖定位(chain `scratchpad/lbf_cliff_chain.sh`)

| arm | noRT | 備註 |
|---|---|---|
| MAX=3.0 | **1.3230** | 預算只 −14%,斷崖 +0.19 全額出現 |
| MAX=2.5 | 1.3345 | |
| MAX=2.0 (DM off) | 1.3422 | vs DM on 1.3524:reserve 白扣 −0.010 |
| ctrl35 rep3 | 1.1328 | 三 control σ≈0.002 |

**定位:斷崖緣正好在 budget=3.0 = `column_sa_legalizer.legalize_rectangles` 的硬閘門
`budget > 3.0` 才進 `_parallel_solve`**(24-worker 平行 restart portfolio + 唯一會消費
direct/flow `sample_fn` 的路徑;early-exit 時間解剖獨立發現同一結構,見
`docs/design/2026-08-04-early-exit-true-time-reduction.md`)。MAX≤3.0 時整個 tail 退化成
單執行緒鏈,失去 24× 平行寬度與全部 ML 通道——這不是 SA 收斂彈性,是結構性截斷。

推論:先前「direct-seed 救援無效」(MAX=0.5+DM0.05 wash)的結論**無效**——閘門之下
`sample_fn` 根本沒被消費過,種子從未進場。pool-gate 打開後需重測。

## 實驗 3:`PARTNER_POOL_GATE`(patch + chain `scratchpad/lbf_poolgate_chain.sh`)

Patch(`partner/column_sa_legalizer.py` 兩處,default bit-exact):
1. `budget > 3.0` → `budget > PARTNER_POOL_GATE`(env,default 3.0)。
2. worker margin `deadline_A − 0.30` → `deadline_A − min(0.30, 0.15·remaining)`
   (remaining ≥2s 時數學上恆等於 0.30,舊路徑 bit 級不變)。

結果(chain `bnlamb000`,全臂 100/100 feasible):

| arm | noRT | avg_rt | vs 閘門關 | 備註 |
|---|---|---|---|---|
| pg_ctrl35(gate 3.0)| 1.1301 | 1.99s | — | **中性驗證過**(ctrl 家族 1.1284–1.1328) |
| pg_max20 | **1.2884** | 1.23s | −0.064 | |
| pg_max10 | **1.3044** | 0.79s | −0.081 | |
| pg_max05 | **1.3515** | 0.41s | −0.072 | |
| pg_max025 | **1.4056** | 0.23s | −0.083 | |
| pg_max05_dm | 1.3681 | 0.67s | +0.017 vs pg_max05 | **direct 低預算救援正式判死**:更差且 GPU 延遲爆牆鐘(exP50 +0.18) |
| pg_max35_full | 1.1316 | 1.92s | wash | b<60 1.352→1.223、b60-99 1.180→1.157 大improve,λ 權重下 total 不動;tail +0.0075(邊界噪音) |

結論:
1. **pool 寬度佔斷崖 ~30%**(−0.06~−0.08 全檔位收復),`PARTNER_POOL_GATE=0` 在低預算檔全面優於關閘,無任何檔位退化 → 低預算 regime 促轉。
2. pool 路徑實際牆鐘低於名目預算(margin 提前返回,exP50 −0.33@2.0):**avg runtime 軸的真實前緣 = (1.23s→1.288) / (0.79s→1.304) / (0.41s→1.352) / (0.23s→1.406)**。
3. 殘餘 +0.16(3.5 vs 2.0)未解。新假說=**direct-refine ladder(NREF slots,tail winners 主源)首輪 rung 需 ~3s**:3.5s 剛好跑完(→24s 平坦),2.0s 跑不完。

## 實驗 4:ladder 假說定罪(chain `lbf_ladder_probe.sh`,100/100 feasible 全臂)

| arm | noRT | 判讀 |
|---|---|---|
| ctrl35_dmoff(3.5,direct 全關)| **1.2563** | **direct ladder 在 3.5s 值 −0.126** —— 全解算器最大單一品質組件 |
| pg2_dmoff(2.0 gate0,direct 關)| 1.2875 | ≈ pg2_rep2(1.2844):**2.0s 時 ladder 淨值 ≈ 0**(rung 不完 → 15 slots 白佔) |
| pg2_nref6(2.0 gate0,NREF=6)| 1.2880 | slot 重配於 2.0s 無感 |
| pg2_rep2 | 1.2844 | pg_max20 rep 內 σ≈0.002 |

**斷崖完整解剖(3.5→2.0 = +0.222)**:pool 寬度 ~0.064(已收復)+ **ladder 價值消失
~0.126(現行主標的)** + 真 SA 彈性 ~0.03(kernel 吞吐可攻)。攻擊面:把
`_worker_refine` 的 direct-prediction 精修做成 anytime(價值/秒排序,任何截止點都交得出
競爭解)± rung 加速(numba kernel)。若拿回 −0.12,前緣預估:2.0s→~1.16、0.25s→~1.28。

## 實驗 5:PARTNER_EARLY_EXIT 探針(chain `ee_probe_chain.sh`;EE 實作 = cherry-pick
`af930b3`,設計 `docs/design/2026-08-04-early-exit-true-time-reduction.md`)

| arm | noRT | avg_rt | 判讀 |
|---|---|---|---|
| ee_ctrl35 | 1.1324 | 1.99s | control(家族帶內)|
| ee_on35 | 1.1368 | **1.79s** | **EE 純減時:−10% 牆鐘只付 +0.0044** |
| ee_on50 | **1.1204** | 2.32s | EE+MAX↑ 再消費:tail 時間饑渴再證(E2 重現);與目標反向,收案 |
| ee_on35_pg0 | 1.1417 | 1.90s | **tail +0.018 退化 → straggler 滲漏假說**:gate=0 讓中小案派 pool 工作,逾期散兵滲入下一案吃 CPU(pg_max35_full tail +0.0075 同號)。gate=0 僅用於低預算檔;3.5 default 維持 gate=3.0 |

EE 判定:同 MAX 下作純減時器可用(−10%/+0.004);「返還再消費」在 tail 是正品質但負
runtime,與本目標(1.000/0.2s)反向。EE 的主戰場改為低預算操作點(見實驗 6)。

## 實驗 6:0.25s 目標檔位精修(chain `op025_chain.sh`)

| arm | noRT | avg_rt | 判讀 |
|---|---|---|---|
| pg025_rep2 | 1.3941 | 0.23s | rep1 1.4056;本檔位 σ≈0.006 |
| pg025_ee | 1.3922 | 0.23s | **EE 低檔位無感**(worker 鏈 ~0.1s,stall 窗觸發不了)|
| pg025_noflow | 1.4047 | 0.23s | **no-flow 判負 +0.011**:flow 種子在 0.25s 仍值回時間成本,留用 |
| pg05_ee | 1.3534 | 0.41s | 0.5 檔 EE 同樣 wash |

## 實驗 7:SA kernel 分數轉換(chain `kernel_score_chain.sh`)

| 檔位 | off | on | Δ |
|---|---|---|---|
| 3.5s | 1.1289 | 1.1311 | +0.002 wash(SA 已收斂,吞吐不缺)|
| 1.0s(gate0)| 1.3044 | **1.2768** | **−0.028** |
| 0.25s(gate0)| 1.3990 | **1.3513** | **−0.048** |

`PARTNER_SA_KERNEL=numba`(commit 3b7ccec):CSR 扁平化 + njit,bit-exact(36 tests 全
`==`),moves/s ~2.7×。冷 JIT ~10s(cache 後 182ms)→ 促轉前需 warm-in-parent 設計。

## 實驗 8:預算曲線形狀重塑(chain `shape_scan_chain.sh`;λ∝e^(n/12) 尾主宰)

| arm | 曲線 | noRT | avg | tailQ | rt_tail |
|---|---|---|---|---|---|
| sh_flat025 | 平 [0.15,0.25] | 1.3508 | 0.247 | 1.3575 | 0.23 |
| sh_tau8 | 9.3e-7·e^(n/8) [0.05,0.8] | 1.3043 | 0.240 | 1.2905 | 0.56 |
| sh_tau10 | (SCALE 誤配,全體餓死)| 1.4393 | 0.162 | 1.4447 | 0.18 |
| **sh_tau12** | 6e-5·e^(n/12) [0.05,1.0] | **1.3002** | 0.237 | **1.2852** | 0.54 |

**小案讓路、tail 加菜:−0.051**。tau10 誤配臂反證 tailQ 對 tail 預算的劇烈敏感。
新地板浮現:小案牆鐘 ~0.15-0.17s(初判串行頭,後由 `[ee]` 解剖推翻:pre=0.003,
**地板在 solve 內部** = pool 編排 / 串行建構的固定成本)。

## 實驗 9:pool-floor 路由(chain `poolfloor_scan.sh`)判死

小案改走串行路(GATE=0.1/0.2)只省 0.03s/案(串行地板也 ~0.12s),tail 微增益被小案
品質損失吃掉:pf_g01 1.3104 / pf_g02 1.3047 vs pf_ctrl 1.2941(tau12 rep,tier σ≈0.006)。
**GATE=0 維持目標檔正解**;~0.12s 兩路共同固定地板留作後續(期望 ~−0.02)。

## 實驗 10:PARTNER_ANYTIME_LADDER(commit bceac97)A/B — 全檔位 wash/敗,不促轉

Agent 診斷(單執行緒 proxy 證據堅實):ladder 四缺陷疊加(全有全無交付、`_tighten`/`run`
尾段餓死、壞順序、`legalize_soft` 無界 3× 超時),且 **promoted 3.5s 檔 n≳110 段 15 個
refine slot 本已 100% 浪費**(rung 全失敗)。proxy n=116@1.7s 1.535→1.183。

Full-100 成對判定:
- MAX=2.0+gate0+kernel:ctrl 1.2479 / on 1.2599(**+0.012 敗**;pool 競爭下 rung 仍不完,
  寬鬆候選擠占選擇)——順帶確立 **kernel@2.0 −0.038**(1.286→1.248)
- MAX=3.5 三輪成對:−0.0084/+0.0047/+0.0019,**均值 −0.0006 wash**(rep1=方差)
- 目標檔 direct 復活(AL+DM0.3+NREF6):1.3000 ≈ 家族(wash)

結論:anytime 重排在單執行緒下機制為真,但競爭 regime 的 rung 完成時間才是根因。
**次級假說:把 ladder 熱迴圈(`legalize_soft`/`_tighten`)kernel 化(SA kernel 只覆蓋
SA 移動迴圈),rung ÷2.5 後 AL 前提才成立**。代碼留存 default off。

## 實驗 11:PARTNER_REFINE_KERNEL(0805,commits ca9adea/015fe0f)——雙檔位突破

Rung(`legalize_soft`)÷12-23、`_tighten` ÷12-14、`_Refiner` build 6-11×(FASTBUILD),
bit-exact(78 tests、off 路徑 byte-identical)。真熱點 = Python 層衍生結構迭代
(_evict 35%、edge-list 25%),非重疊偵測(8%)。warm 在 pool fork 前(children 繼承)。

| arm | noRT | avg | tailQ | 判讀 |
|---|---|---|---|---|
| rk_ctrl35 / rk_on35 | 1.1317 / **1.1208** | 1.99 / 1.88s | 1.1192 / 1.1094 | rep1 **−0.0109** |
| rk_ctrl35_r2 / rk_on35_r2 | 1.1275 / **1.1184** | 1.99 / 1.88s | 1.1172 / 1.1059 | rep2 **−0.0091**(符號一致,promoted 檔歷史新低;runtime 同降)|
| rk_goal_ctrl / rk_goal_on | 1.3115 / 1.3217 | 0.214 / 0.199s | — | RK 單獨在目標檔微負(價值在 direct 消費)|
| **rk_goal_dm03**(RK+DM0.3+NREF6)| **1.2610** | **0.218s** | **1.2340** | **direct 通道在 0.44s tail 復活:−0.050,n≥110 帶 −0.074** |

戰略含義:「通道不消費先驗」的 binding constraint 翻轉——prior 品質開始承重,
訓練 gate(docs/research/2026-08-05-cost-finetune-fewstep-survey.md)前提成立。

## 現況總結(0804 晚)

- **目標檔(avg≤0.24s)最佳組合 = kernel + gate0 + tau12 曲線:noRT 1.294-1.300**
  (開場 flat-0.25 無 kernel 1.489 → −0.19)。精確壓 avg 0.2(tail 0.55→0.50)估 ~1.30。
- 全前緣:0.24s→1.294 / 0.78s→1.277(kernel)/ 現役 3.5s 1.129-1.133 @ 2.0s。
- 目標 (noRT 1.000, avg 0.2s) 殘距 ~0.30。剩餘攻擊面:ladder anytime(agent 進行中;
  tail 0.55-0.8s 預算下實際可收復量待 A/B)、restart 廣度(configs 只有 24 條,
  POOL=46 需程序化擴列)、~0.12s 固定地板(期望 −0.02)、GPU 空轉臂、
  最後是 prior/拓撲品質牆(0707 定律域)。

## 實驗 12(0805):目標檔 G3-G5 旋鈕收官 — st4 促轉,組合不疊加

- **方差牆確立**:combo 家族帶 1.227-1.257(σ≈0.009)。G3「全臂皆勝」是 control 漂移
  (G4 反例:同臂 st6 兩鏈 1.2287/1.2525);此檔位單 run 不可判 <0.01 的效應,
  促轉一律成對多輪。
- **g4/g5 判定:`PARTNER_DDIM_STEPS=4` 促轉**(兩清潔成對 −0.0116/−0.0126,tailQ
  同向;st10→8→6→4 單調——目標檔 binding 是 GPU 時間不是 prior 保真度)。
  組合(st6+flow6+AL)不疊加(≈家族均值);flow6/AL 單項留 default。
- **目標檔定案 config(0805)**:RK + SA kernel + gate0 + tau12(SCALE=5e-5,
  clamp[0.05,0.75])+ DM0.3 + NREF6 + **dpmpp4** + flow st8 →
  **noRT 1.223-1.230 @ avg 0.211-0.217s**(g5_st4_b 1.2234 最佳單點)。
- Session 目標點軌跡:1.489(flat 0.25s 無 kernel)→ 1.223(−0.266)。

## 實驗 13(0805):st 地板與 flow 定案 — 目標檔旋鈕空間收攏

- **g6:`PARTNER_DDIM_STEPS=2` 促轉**(vs st4 成對 −0.0108/−0.0170 符號一致)。
  單調鏈 st10→8→6→4→2 全程成立:目標檔 direct = 粗略拓撲草圖產生器,品質由
  RK 精修扛,GPU 時間釋放一路是淨贏。
- **g7:st1 wash**(+0.0100/−0.0010 符號不一致)——地板在 st2。**flow6 判死**
  (vs flow8 +0.0140/+0.0031 兩對皆正;G3 單 run「勝」= 漂移,方差牆紀律再驗)。
- **目標檔最終定案 config**:dpmpp2 + flow st8 + RK + SA kernel + POOL_GATE=0 +
  tau12(SCALE=5e-5, clamp[0.05,0.75])+ DM0.3 + NREF6 →
  **noRT 家族 1.212–1.223 @ avg 0.206–0.217s**(最佳單點 g6_st2_b 1.2121 @ 0.211s)。
- Session 目標點軌跡:1.489 → 1.212(**−0.277**)。旋鈕空間關閉;殘餘結構線 =
  GPU 空轉臂(phase-B 二波,agent 進行中)→ prior/拓撲牆。
