# 08-29 收官計畫:兩條軌(Flow fine-tune round 2、尾帶 runtime 校準)

更新:2026-08-29 16:45 UTC。deadline 2026-08-31 23:59;使用者要求 8/31 重打包前繼續提分。
出貨包不變(`submission/cadc1013_0828b_final.tar.gz`,md5 `751f7e37…`),兩條軌都是「候選 vs 現包」的同鏈配對量測,沒過就照舊出貨。

## 0. 重要發現:runtime factor 在尾帶沒貼 floor(以現包 dry run 7 逐案重算)

評分 `cost = quality × max(0.7, (rt/median)^0.3)`,floor 需 `rt ≤ 0.305 × median`。用 0823 更新版 median(`docs/official/beta_test/C_median_runtimes_beta_hidden_update_20260823.csv`)、本機→contest 時間係數 M=1.45、field 加速係數 D(final median = D × 0823 median):

| 情境 | dry run 7 期望 total | floor(0.7 × raw 1.1009) | 超 floor 損失 | 超 floor 案數 |
|---|---|---|---|---|
| D=1.0 | 0.7973 | 0.7706 | **0.027** | 21 |
| D=0.8 | 0.8355 | 0.7706 | 0.065 | 47 |
| D=0.7 | 0.8650 | 0.7706 | 0.094 | 52 |
| D=0.6 | 0.9028 | 0.7706 | 0.132 | 61 |

損失幾乎全在 tid 90–99(n=111–120,本機 rt 0.81–1.21 s,median 2.2–5.5 s):D=1.0 時每案 factor 0.71–0.79,各損 0.004–0.006。
對照:raw 剩餘缺口到 1.08 只有 0.02;而 `budget_table_mid_m145`(全帶預算 ÷1.45,chain EM 的 emFT 兩 rep)raw +0.04 但 D≤0.8 的期望 total 反而好 0.013–0.035,D=1.0 持平略差 0.005–0.01。
→ **只縮尾帶預算**(其他帶已貼 floor、縮了純虧)理論上在所有 D 都不虧、D<1 大賺;代價是 raw 上升(估 +0.01–0.015)。這與 08-27 裁定的「raw 進步且 runtime-aware 不退」相衝(raw 會退),**需要使用者裁定目標函數**:排名用 total(含 runtime factor),不是 raw。

**M 的注意事項**:現包的 solver 是 wall-clock 預算制(SA/refine 到 deadline 就停),所以 contest 機上的 runtime ≈ 同樣的秒數(只有品質變差),不是 ×1.45;M=1.45 是 beta 那個 CPU-torch 壞包反推的(sampler 1 s/案不受預算約束),而 §17ae 用同一壞包逐帶重放顯示 contest 機尾帶 1.53 s vs 本機 1.90 s(M≈0.8,本機有 load)。現包合理的 M ≈ 0.9–1.2(early-exit 在慢機上較晚觸發會把 rt 推向滿預算)。M=0.9 時:D=1.0 尾帶剛好貼 floor(R≈0.26)、D=0.8 損 ≈0.015、D=0.7 ≈0.04、D=0.6 ≈0.07。結論不變:要不要縮尾帶 = 對 D(final field 相對 0823 median 的加速)的押注,`rt_pairs.py --M 0.9` 與 `--M 1.45` 兩種都要看。

工具:`scratchpad/rtaware/rt_total.py`(單檔 runtime-aware 總分)、`scratchpad/rtaware/rt_pairs.py`(配對 + D 先驗期望值)。四套 shadow 與 official 的 n↔test_id 對映完全相同,median 按 test_id 套用皆有效。

## 1. 軌 A 結果:尾帶 budget 校準(四套 ×2 同鏈反序,現包 env;16:20–17:00 UTC,load 14–20)

候選表(只動 n≥105,`scratchpad/rtaware/budget_table_k{20,25}.txt`):`budget_n = min(mid_n, κ × median_tid(n))`。
k20(κ=0.20):n=112–114/117–120 縮 23–41%,其餘 ≤12%;k25(κ=0.25):只縮 n=113/114 24–26%、117–119 13–18%。

| 套 | base raw | k20 raw(Δ) | tail rt base→k20 | total(M=1.0)Δ @ D=1.0 / 0.9 / 0.8 / 0.7 / 0.6 | total(M=1.45)Δ @ D=1.0 / 0.7 |
|---|---|---|---|---|---|
| official | 1.0984 | 1.1063(+0.008) | 0.880→0.757 | +0.005 / +0.003 / −0.003 / −0.013 / −0.024 | −0.015 / −0.027 |
| v3 | 1.1331 | 1.1428(+0.010) | 0.886→0.727 | +0.007 / +0.004 / −0.003 / −0.014 / −0.030 | −0.016 / −0.034 |
| v5 | 1.1010 | 1.1010(±0) | 0.873→0.753 | ±0 / −0.002 / −0.006 / −0.017 / −0.031 | −0.019 / −0.034 |
| v6 | 1.1256 | 1.1310(+0.005) | 0.881→0.778 | +0.004 / +0.001 / −0.004 / −0.012 / −0.021 | −0.013 / −0.023 |

k25:raw +0.003~+0.010 但幾乎不省時間,被 k20 支配,淘汰。
判讀:k20 = 押 field 加速。M=1.0 時損益兩平在 D≈0.85;D=1.0 最多虧 0.007,D=0.7 賺 0.012–0.017,D=0.6 賺 0.02–0.03;M≥1.2 時全 D 都賺。raw 退 +0.005~+0.010(v5 持平),全部落在 105+ 帶(設計如此)。
**待使用者裁定**(排名用 total 還是 raw;對 D 的信念)。裁定前不入包。四臂確認鏈(§3)已把 k20 與 FT2 一起排進去,裁定時證據會是 ×4。

結果檔:`artifacts/shadow/rt{Base,K20,K25}_{r1,r2,v3r1,v3r2,v5r1,v5r2,v6r1,v6r2}.json`;log `~/.claude/jobs/06af0e53/tmp/chainRT*.log`。

## 2. 軌 B:Flow fine-tune round 2(GPU0,進行中)

deep-reasoner 設計(`scratchpad/flow_ft_0829/STATUS.md`):從 round-1 300k EMA 續訓,tilt 溫度 T=24→**12**(= evaluator 權重 exp((n−120)/12) 的精確重要性抽樣;尾帶抽樣占比 0.50→0.74),lr 1e-5、warmup 1k、cosine 退火到 0.01×、250k 步、EMA 0.9998,目標函數與 v1 bit-exact。
- 進度:`artifacts/flow_ft_0829/train.log`;ckpt `artifacts/flow_ft_0829/flow_ft0829_ft300k_tailT12_lr1e-5_wu1k_bs12_s250k/`(每 25k 一個 `step_*.pt`,250k 完成)。
- 吞吐 5–15 it/s(與 evaluator 鏈搶 CPU),ETA 08-29 21:45 – 08-30 04:00 UTC。
- Gate 鏈已排隊:`scratchpad/rtaware/chainFT2.sh`(等 `step_00250000.pt` → `export_ema_only.py` → preflight 檢查 step=250000 → 四套 ×2 反序,base = 現包 300k EMA)。結果 `artifacts/shadow/ft2{Base,T12}_{r1,r2}_{off,v3,v5,v6}.json`。
- 陷阱:ckpt 目錄裡的 `final.pt` 目前是 200 步 smoke,不可匯出;續跑不可改 `--lr`。
- 判讀:四套配對 Δ + 分帶 Δ + n=76–89 的 column 出貨數(暴增 = §17t 崩塌重現 → 否決)。

## 3. 四臂確認鏈(已排隊,`scratchpad/rtaware/chainFinal.sh`,log `~/.claude/jobs/06af0e53/tmp/chainFinal.log`)

等 round-2 `step_00250000.pt` → EMA-only 匯出 → preflight → 四套 × 四臂 × 4 reps、臂序輪換(ABCD / DCBA / BDAC / CADB):
A = 現包 env(ft0828 300k EMA + mid 表)、B = FT2(T12 250k EMA)、C = k20、D = FT2 + k20。結果 `artifacts/shadow/fn{A,B,C,D}_r{1..4}_{off,v3,v5,v6}.json`;鏈尾自動印 `analyze_pairs.py`(B/C/D vs A,每套)。k20 臂另用 `rt_pairs.py --M 1.0/1.45` 看 total。預計 08-30 00:30–02:00 UTC 完成。

## 4. 結果(08-30 00:10 UTC;細節 experiments doc §18d)

FT2 促轉:official −0.001(wash)、v3 −0.019、v5 −0.014、v6 −0.016(4 reps 配對,兩套 CI 不含 0),runtime 持平,mid band 無崩塌。
k20 疊 FT2:raw +0.002~+0.009,total(M=1.0)D=1.0 wash、D=0.8 −0.008、D=0.7 −0.019、D=0.6 −0.032;M=1.45 −0.03。**使用者裁定 B(FT2)或 D(FT2+k20)。**
候選包(已複製到 repo):B = `submission/cadc1013_0830_ft2_final.tar.gz` md5 `b7f73fb2eae123d3b4b3ce76a380f4cf`(dry run 8:1.0958、100/100、avg 0.364 s、max 1.22 s、守門 0);D = `submission/cadc1013_0830_ft2k20_final.tar.gz` md5 `0ec983a4a6681cf80857447f34272629`(dry run 9:1.0984、100/100、avg 0.347 s、max 0.96 s、守門 0)。逐案 JSON `artifacts/shadow/dryrun{8_ft2,9_ft2k20}_off.json`;log `~/.claude/jobs/06af0e53/tmp/dryrun{8,9}.log`。工作樹目前 = B 設定(`apply_pack_variant.sh ft2`);要 D 就 `apply_pack_variant.sh ft2 k20` 再 `pack_cadc1013.sh`。

## 5. 08-30 追加結果(細節 experiments doc §18e–18j)

- 預算極限:純加時間極限 ≈1.07(×30);候選寬度才有效 → `FLOW_SLOTS=16 / NREF=12`(s16)四套 ×4 promote(off −0.010 / v3 −0.006 / v5 −0.004 / v6 −0.011,runtime 持平)。
- 殘餘 violations 為結構性(locked 牆線、packed cluster;離線探針 0 修復)→ 不再開。
- 守門觸發根因(MIB unification 面積異質)→ 守門加面積修復 + raw column fallback;MIB 修法不入包。
- 最終候選包:B′ `submission/cadc1013_0830_ft2s16_final.tar.gz`(md5 2ee934a9…,dry run 10:1.0916)、D′ `submission/cadc1013_0830_ft2k20s16_final.tar.gz`(md5 dd4198c8…,dry run 11:1.0861、max rt 1.07 s)。工作樹 = B′ 設定(`apply_pack_variant.sh ft2 s16`)。

## 6. 收尾順序

1. 軌 A 官方/v3/v5/v6 結果出來 → 把「raw vs 各 D 期望 total」表交給使用者裁定(要不要接受 raw 退、total 進)。
2. 軌 B gate 出來 → 若四套同向且 mid band 不崩,FLOW_CKPT 換 round-2。
3. 任何入包組合必須「完整候選 env vs 現包 env」同鏈 ×4 兩序(§17y 鐵律)後才重打包(`scripts/pack_cadc1013.sh` → `scratchpad/gate_chains_0828/dryrun7.sh` 流程)。
4. 都沒過 → 8/31 照 0828b 演練上傳。
