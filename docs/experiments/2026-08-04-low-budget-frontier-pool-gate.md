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
