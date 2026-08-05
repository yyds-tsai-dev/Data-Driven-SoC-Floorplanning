# PARTNER_GPU_ARM — 把空轉 GPU 變成 phase-B 二波候選供給

- 狀態:**default off**,off 路徑 bit 級不變(單元測試 39 綠佐證)
- 旗標:`PARTNER_GPU_ARM=1`(+ `_TARGET` / `_MIN_A` / `_K` / `_TS0` / `_SEED` / `_DEBUG`)
- 代碼:`partner/column_sa_legalizer.py`(模組頭輔助函式 + `_parallel_solve`)、
  `partner/contest_optimizer.py`(sampler 的 `gen_seed` 偏移)
- 測試:`tests/test_partner_gpu_arm.py`

---

## 1. 問題:加速器在整個 CPU 期間空轉

兩個操作點的 case 內時間結構:

| 檔位 | 曲線 | worker span | GPU 忙碌 | GPU 空轉 |
|---|---|---|---|---|
| promoted | `0.06·e^(n/20)`,`BUDGET_MAX=3.5` | ~3.2 s | 0.1–0.3 s | **~2.9–3.1 s** |
| goal | `6e-5·e^(n/12)`,`[0.05, 1.0]`(`DIRECT_MIN=0.3`) | ~0.44 s | ~0.1 s | **~0.34 s** |

GPU 只在 case 開頭抽一批 direct/flow 候選(`sample_fn(n_ref)`),之後整段
column SA + refine 都是純 CPU。本地 L4 已如此,官方 A100 更快 → 空轉比例更高。

`_parallel_solve` 的既有時間軸:

```
t_pb                                                              deadline
 |── map_async(_worker_solve, configs)  ← 全部 column slot 派工
 |── sample_fn(n_ref)   [GPU 0.1–0.3 s]
 |── map_async(_worker_refine, ref_payloads)
 |······································ 主進程阻塞在 res.get() ······|
                                              ↑ GPU 從這裡到 case 結束全空
```

## 2. 現行 phase-B 結構(為何它一直是死碼)

`PARTNER_PHASE_B`(fraction,default 0)把 case 預算尾段切出來,收回 workers 後
對 phase-A 贏家做 8 個 seed/v_weight/anchor 變體 re-refine。三個閘門讓它從未上場:

1. `PARTNER_PHASE_B_MIN_BUDGET=8` — 兩個檔位(3.5 s / 0.44 s)都構不到;
2. 進場守衛 `time.time() < deadline - 1.5` — 0.15 s 的 slice 永遠過不了;
3. `wd2 = deadline - 0.30` — 低預算下 worker deadline 直接落在過去。

另外一個**未被利用的事實**:phase-B 只派 8 個 payload,而 pool 有 24–46 個
worker。**phase B 今天是嚴重 under-subscribed 的**,新增候選的 CPU 邊際成本是零。

歷史備註:serial 版(vkill stage2)輸給 SA 自己的尾段;八個 **concurrent**
basin-hop 是不同命題,且從未在 RK 之後測過。

## 3. RK 翻轉的經濟學

`PARTNER_REFINE_KERNEL=numba`(0805 促轉)把 refine rung(`legalize_soft`)
從 0.36–0.8 s 壓到 ~0.035 s(÷12–23,bit-exact)。因此:

- 0.15 s 的 phase-B slice 現在裝得下 ~4 個 rung(以前 0 個);
- 3.5 檔的 0.8 s slice 裝得下 ~20 個。

`phase_b_min_budget_default()` 因此隨 RK 開關自適應:RK on → 0.35 s,
RK off → 1.5 s,**arm 關閉時維持歷史 8.0**(不動既有 lever 的語意)。

## 4. 設計:GPU 空轉臂

```
t_pb                          deadline_A                        deadline
 |── phase A:全部 column slot + 既有 NREF refine slot(照舊)──|── phase B ──|
 |── wave 1  sample_fn(n_ref, gen_seed=0)      [測量 t_s1]
 |                       └── wave 2  sample_fn(K, gen_seed=8117+…)
 |                            (在 phase A 期間,延遲被 CPU workers 遮蔽)
                                                              ├ 8 個贏家變體
                                                              └ K 個新 GPU 候選
                                                                 (RK 加速 refine)
                                                              → 同一個 true-cost 選擇
```

**關鍵區別**(相對於已判死的近親):

- **不搶 phase-A 的 24 個 column slot**。這是時間軸切片,不是 slot 重分配 —— 與
  `PARTNER_NREF 10→15`(goal tier +0.048,蠶食 column 廣度)本質不同。
- **不擋 case 起跑**。0723 判死的「開頭就採樣」是把 GPU 延遲加在 case 前端;
  wave 2 發生在 phase A 執行期間,workers 一直在跑,延遲被遮蔽。
- **合法性不變**:phase-B 產物走同一個 `score()` 選擇,勝出時 `win_is_ref=True`
  → `_ensure_no_overlap`。選擇端只接受合法解。

### 兩檔位時間線(數值)

promoted(rem = 3.2 s,est ≈ 0.20 s):

```
0.00 ─ 派工 24 column configs
0.00 ─ wave 1 (0.20 s)  → 8 refine payload 派出
0.20 ─ wave 2 (0.20 s)  ← 遮蔽在 phase A 內
2.40 ─ deadline_A;收回 workers,選出 win
2.40 ─ phase B  slice = 0.80 s:8 變體 + 12 個 GPU 候選 = 20 payload(pool 24)
3.20 ─ deadline
```

goal tier(rem = 0.44 s,est ≈ 0.10 s):

```
0.000 ─ 派工
0.000 ─ wave 1 (0.10 s)
0.100 ─ wave 2 (0.10 s)?  ← 由 `gpu_arm_second_wave` 的 fit 檢查決定
0.286 ─ deadline_A
0.286 ─ phase B  slice = 0.154 s:8 變體 + 4 個 GPU 候選
0.440 ─ deadline
```

goal tier **本來就是邊際的**:`rem − slice − 2·est = 0.44 − 0.154 − 0.20 = 0.086`,
低於 `MIN_A·rem = 0.154` → 在 est = 0.10 s 時 **arm 主動放棄 carve**,路徑退回
今天的行為。只有當實測 sampler 延遲 ≲ 0.065 s 時 goal tier 才會進場。這是刻意的:
carve 必須被賺到,不是被假設。

## 5. K / frac 公式與理由

### phase-B slice(= frac × rem)

```
slice = clamp(0.20·rem, 0.35·rem, PARTNER_GPU_ARM_TARGET=0.80)
frac  = slice / rem
```

- **絕對目標 0.80 s**:3.5 檔能負擔而不掏空 column 深度(≈20 個 RK rung)。
- **35 % 上限**:讓 0.44 s 的 goal tier 仍拿得到 ~0.154 s(≈4 個 rung);
  再高就把 phase A 砍到不成樣。
- **20 % 下限**:長預算下不讓絕對目標把 slice 壓成無關緊要的零頭。

代入:rem=3.2 → 0.80(frac 0.25);rem=0.44 → 0.154(frac 0.35);rem=24 → 4.8(frac 0.20)。

### carve 准入(反「白扣」守衛)

```
carve  ⟺  rem ≥ phase_b_min
      ∧  rem − slice − 2·est  ≥  PARTNER_GPU_ARM_MIN_A(0.35) · rem
```

`est` = 該 case block-count **十位帶**的 wave-1 延遲 EMA(`_GPU_ARM_TS`),
首次以 `PARTNER_GPU_ARM_TS0`(0.25 s)為先驗。這是 **可重用的 instance 統計
(block count band)**,不是 case id —— 符合 CLAUDE.md 的鐵律。

語意:付掉兩波採樣後,phase A 仍必須保有整體預算的 35 % 作為**真正的 SA 時間**。
0723 `dmoff` 的「budget carve 白扣 −0.010」就是缺這一條。

### 二波候選數 K

```
K_appetite = clamp(0, min(pool, PARTNER_GPU_ARM_K=12), 2 + ⌊slice / 0.06⌋)
(K, n_var)  = split(K_appetite, pool):
     pool 容得下 K + 8 → 不動變體(production pool 24–46 恆成立)
     否則 K ≤ pool//2,變體降到 pool − K(incumbent 至少拿一半)
```

- `2 + slice/0.06`:一個候選約對應 60 ms 的 slice(raw prediction 比贏家變體
  需要更多 rung 才收斂);slice 0.154 → K=4,slice 0.80 → K=15 → 被 cap 12。
- **pool 上限**:phase B 今天只用 8/24 個 slot,K=12 落在閒置 slot 上 → CPU 邊際成本 0。
- **`_K` cap 12**:採樣批次延遲隨 batch 增長,必須塞得進 phase A;
  且 `PARTNER_KS_CAP` 已在 sampler 內另有把關。

### wave-2 fit 檢查(執行期)

```
launch  ⟺  now + 1.20·t_s1 + 0.03  <  deadline_A
```

用**剛剛實測**的 wave-1 延遲 `t_s1`(比 EMA 更準;batch 有 cap,步數主導延遲,
所以 wave 2 ≈ wave 1)。不過就不發,phase B 退化成純變體輪。

### 生成器種子

sampler 的 `gen` 是釘死的(`manual_seed(17)` direct / `23` flow),同參數再呼叫
一次會**原樣重抽 wave 1**。因此 `sample_fn` 增加 `gen_seed`(偏移,default 0
→ 與已審查基線 byte-identical),legalizer 以 `inspect.signature` 檢查
sampler 是否接受該關鍵字 —— **legacy 一參數 sampler 一律拒發 wave 2**,
不會把 carve 花在重複批次上。

`PARTNER_FLOW_ZORDER` / `PARTNER_FLOW_NOPT` 這兩個 opt-in 子採樣器仍用自己的
釘死種子(兩個檔位都關著);為防萬一,legalizer 對 wave 2 做一次與 wave 1 的
陣列去重。

## 6. 一併修掉的低預算 bug(僅在 auto carve 路徑生效)

| 位置 | 舊值 | auto carve 下 | 理由 |
|---|---|---|---|
| phase-B 進場守衛 | `deadline − 1.5` | `deadline − 0.5·slice` | 0.15 s slice 否則永遠進不去 |
| phase-B worker margin | `deadline − 0.30` | `deadline − min(0.30, 0.15·remaining)` | 0.30 s 會把 worker deadline 推到過去(與 3115–3118 行 `worker_deadline` 同一個慣用法) |

手動 `PARTNER_PHASE_B=…` 路徑兩者都維持原值 → 既有 lever byte-identical。

## 7. 收益上限估計(誠實版)

沒有 A/B 之前這只能是量級估計,不是預測:

**promoted 3.5 檔.** phase B 新增 12 個「來自不同噪聲種子的 direct 候選 ×
0.8 s RK refine」。同檔位的可比證據點:`PARTNER_NREF` 把 direct slot 從 8 加到
15 在 n≥95 有效(已在 promoted env);exp 11 顯示 **RK 的價值主要在 direct 通道
被消費**(3.5 檔 −0.0107)。二波候選是 direct 通道供給的 +150 %(8 → 20),
但拿到的 refine 時間只有 phase-A slot 的 ~25 %(0.8 s vs 2.4 s+),且要扣掉
phase A 縮短 25 % 的損失。**上限估計 −0.005 ~ −0.015,期望值靠近 0**;
最可能的結果是 wash。判準因此設在 −0.005。

**goal tier.** 依 fit 檢查,多半**根本不進場**(見 §4 數值)。若實測 sampler
延遲夠低而進場,收益來源是 `rk_goal_dm03` 已證的「direct 通道在 0.44 s tail 復活
(−0.050,n≥110 帶 −0.074)」之延伸 —— 供給再翻倍。但 phase A 只剩 0.286 s,
column restart 深度損失可能吃掉全部。**這一檔是真正的 coin flip。**

**風險(按嚴重度)**

1. **phase-A 縮短的白扣**(主要風險)。緩解 = §5 的 carve 准入守衛;
   殘餘風險 = 守衛用的是延遲估計,不是品質估計 —— 它保證「兩波塞得下」,
   不保證「新候選比損失的 column 深度值錢」。**只有 A/B 能判。**
2. **raw prediction 在短 slice 內 refine 不完**。不會產生非法解(選擇端把關),
   但會浪費 slot;`K ∝ slice` 就是在對沖這一點。
3. **`_GPU_ARM_TS` 帶來的跨 case 順序相依**:第 k 個 case 的 carve 決策取決於
   同 block 帶更早的 case。每個帶跑過一次後就收斂;不是 case-id 綁定,但
   重跑時前幾個 case 的行為可能不同。已記錄,非缺陷。
4. **官方 RuntimeFactor**:arm 不加時間(carve,不外加),per-case 牆鐘不變。

## 8. 單案 micro-run(機制驗證,**不是**品質證據)

`scratchpad/gpuarm_probe.bash`,test_id 95(n=116),`PARTNER_POOL=8`,
`BUDGET_MAX=3.5`,RK on,真實 direct checkpoint(`eval_step1p2M.pt`):

```
[gpuarm] gate n=116 rem=3.422 est=0.250 frac=0.234 auto=1 phase_b=1
[gpuarm] n=116 rem=3.422 frac=0.234 slice=0.800 est=0.250 auto=1 wave2=4 payloads=8
```

- carve 依公式取到 `slice = 0.800 s`(frac 0.234 = 0.80/3.422)✓
- `est=0.250` 是 `TS0` 先驗(單案跑不出 EMA)✓
- pool=8 → split 成 4 變體 + 4 GPU 候選 = 8 payload(production pool 24 會是 8+12=20)✓
- feasible 1/1,runtime 3.30 s(off 控制 2.96 s),兩者都在 3.5 s 預算內 —— **arm 不加時間**

**品質數字刻意不列為證據**:同一個 on 設定兩次分別得 1.1867 / 1.0835,off 得 1.1794。
單案 wall-clock-bounded 解的方差(~0.10)遠大於任何真實效果。這次 micro-run 只證明
「機制會跑、閘門如設計動作、解合法、時間不外溢」。

## 9. A/B 計畫

成對(paired)全 100 案,由排程者跑;**本 worktree 不跑多案評測**。

| 檔位 | 控制組 | 實驗組 | 促轉判準 |
|---|---|---|---|
| goal | tau12 + gate0 + RK + DM0.3 + NREF6 | 同 + `PARTNER_GPU_ARM=1` | ΔnoRT ≤ **−0.01** vs 1.243 ± 0.009 家族 |
| promoted 3.5 | 現役 .env(RK on) | 同 + `PARTNER_GPU_ARM=1` | ΔnoRT ≤ **−0.005** vs 1.12 家族 |

**先跑診斷再跑判定**:兩檔位各跑一輪 `PARTNER_GPU_ARM_DEBUG=1`,確認

- `auto=1` 的 case 比例(goal tier 若接近 0,先掃 `PARTNER_GPU_ARM_MIN_A`
  0.35 → 0.25 / 0.20,再判定;否則判準測到的是「沒開」);
- `wave2=` 的候選數是否如公式預期;
- `est=` 收斂到多少(這決定 goal tier 能不能進場)。

**掃描順序(若首輪 wash)**:`_MIN_A`(准入鬆緊)→ `_TARGET`(slice 長度)
→ `_K`(候選數)。三者互相糾纏,一次只動一個。

**判死條件**:兩檔位都 wash 或退步,且 `_MIN_A` 放寬後仍無正號一致性 → 
結論寫成「phase-B 二波供給在 RK 之後仍不敵 phase-A 深度」,代碼留 default off。

## 10. 尚未做的(留作後續)

- phase B 仍有閒置 slot 時,用 **elite-2 / elite-3 的變體**填滿(目前只變 winner)。
  刻意不做:會給 A/B 增加混淆項。
- wave 2 目前與 wave 1 同 sampler 同 K 語意;可以改成**條件更強的採樣**
  (例如以 phase-A 贏家為 anchor 的 guided 採樣)—— 那是另一個命題。
- `PARTNER_FUSION` 與本臂互斥(共用同一個 carve);未嘗試合併。
