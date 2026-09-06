# 決賽出貨摘要(給隊友)— 2026-08-29(**08-31 05:20 UTC 上傳定版**)

> **上傳這個檔:`submission/FINAL_UPLOAD/cadc1013.tar.gz`(檔名已正確,直接上傳)。md5 = `3f2cda42c88338fbb32d7670052fc895`。**
> 最終演練(乾淨 py3.13 venv、官方 evaluator):**1.0858、100/100、avg 0.363 s、max 1.31 s、首案 0.050 s、`[selfcheck] cuda_available=True`、flow step 250000、`legal-guard fires: 0`**。QA 合規逐條核對過(A14/A20/A21/A22/A25/A27、48 核/128GB、無絕對路徑、34 檔、requirements 全 pin 含 scipy)。
> 註:md5 與 0830b 的 `6949e99d…` 不同**只因 tar 內 mtime**(apply 腳本重寫過 op_wrapper.py);兩包內容逐檔 diff 完全相同。重新演練若再重組,md5 又會變——**以上傳當下那個檔的 md5 為準**,內容驗證用 `diff -r` 解壓比對。D′(k20)不上傳(48 核專用機 M≈1.0 已貼 floor,k20 只付 raw)。

> **最終候選包(工作樹 = FT2 模型 + FLOW_SLOTS16/NREF12 + 守門面積修復 + runtime 微優化;皆乾淨 Python 3.13 venv 官方 evaluator 演練:100/100、`[legal-guard]` 0 次、Flow step 250000 載入)**
>
> | 包 | md5 | 演練(本機 load 40–55,絕對值偏高)|
> |---|---|---|
> | **B′ `submission/cadc1013_0830b_ft2s16_final.tar.gz`**(建議:raw 優先) | `6949e99d832a40169907ec399f9c08c0` | 1.0914 / avg 0.392 s / max 1.45 s |
> | **D′ `submission/cadc1013_0830b_ft2k20s16_final.tar.gz`**(押 field 加速) | `46ec2c8264c86a28c82f783032c8c255` | 1.1047 / avg 0.359 s / max 1.19 s |
> | 後備 0828b | `751f7e37…` | 1.1009 |
>
> 安靜機期望(×4 gate 平均):B′ public 1.087 / v3 1.111 / v5 1.086 / v6 1.100(三套 shadow 全 <1.12);D′ raw +0.002~+0.009 但 runtime-aware total 在 D≤0.8 賺 0.008–0.03。**B′ 或 D′ 由隊長裁定。**
> 上傳前演練(8/31):B′ `bash scratchpad/rtaware/apply_pack_variant.sh ft2 s16 && bash scratchpad/rtaware/dryrun12.sh`;D′ `... ft2 k20 s16 && bash scratchpad/rtaware/dryrun13.sh`;md5 應一致;看 `[selfcheck] cuda_available=True`、`legal-guard fires: 0`、`feasible=100`;只上傳 `cadc1013.tar.gz`。
> 08-30 全日結論(§18a–18s):對手 1.005 的差距 = 模型 seed 品質(golden 餵我們的 ladder → 1.013);round 3 / round 4 fine-tune 皆 gate 否決 → 模型定案 FT2;runtime 98% deadline-bound,微優化省 4–11 ms/案;殘餘 violations 為結構性。**舊 0830 包(b7f73fb2 / 0ec983a4 / 2ee934a9 / dd4198c8)已被 0830b 取代,不要上傳。**
>
> 以下為 08-30 早上的歷史內容(其 0830 包名與 md5 已過期)。

> (歷史 08:20 版)**08-30 候選包(都在乾淨 Python 3.13 venv 用官方 evaluator 演練:100/100 feasible、`[legal-guard]` 0 次、Flow step 250000 載入)**
>
> | 包 | 內容 | md5 | 演練 noRT / avg rt / max rt |
> |---|---|---|---|
> | **B′ `submission/cadc1013_0830_ft2s16_final.tar.gz`**(建議:raw 優先) | Flow round-2 T=12 250k EMA + FLOW_SLOTS 16 / NREF 12 + 守門面積修復 | `2ee934a99b7e5e3209e9edc156f7ce94` | 1.0916 / 0.374 s / 1.27 s |
> | **D′ `submission/cadc1013_0830_ft2k20s16_final.tar.gz`**(建議:押 field 加速) | B′ + 尾帶預算 k20 | `dd4198c8f2b70cefc4c4fa8ccf6747da` | 1.0861 / 0.348 s / 1.07 s |
> | 後備 `cadc1013_0830_ft2_final.tar.gz` | FT2 only(無 s16、無守門修復) | `b7f73fb2…` | 1.0958 |
> | 後備 0828b | 08-28 版 | `751f7e37…` | 1.1009 |
>
> 四套 ×4 同鏈證據(experiments doc §18d/§18i/§18j):FT2 vs 0828b official wash、v3/v5/v6 −0.014~−0.019;s16 vs FT2 官方 −0.010、v3 −0.006、v5 −0.004、v6 −0.011、runtime 持平;k20 疊 s16 後 official/v3/v5 持平、v6 +0.028,但 runtime-aware total 在 D≤0.8 領先 0.005–0.02。四個完整 env 4-rep 平均:B′ off 1.087 / v3 1.111 / v5 1.086 / v6 1.100;D′ 1.090 / 1.111 / 1.089 / 1.128。
> **上傳哪個由隊長裁定:排名看 raw 就 B′,信 final field 會比 beta 快 ≥20% 就 D′。** 上傳前再演練:B′ `bash scratchpad/rtaware/apply_pack_variant.sh ft2 s16 && bash scratchpad/rtaware/dryrun10.sh`;D′ `... ft2 k20 s16 && bash scratchpad/rtaware/dryrun11.sh`,md5 應一致。
> 守門修復根因:`layout_refiner.py` MIB shape unification 覆寫 soft 成員尺寸不檢查面積,只在「面積異質 MIB 群組」觸發(official/v5/v6 0 個、v3 20 個,hidden 用官方產生器 → 幾乎不會遇到);守門現在失敗時先修面積再走 fallback(合法路徑 bit-exact)。
>
> 以下為 08-29 版原文(0828b 的描述仍有效)。

> 一句話:**上傳 `submission/cadc1013_0828b_final.tar.gz`(md5 `751f7e37e18c603262c15b04a7f4b066`)**,上傳前照第 2 節再演練一次。deadline 2026-08-31 23:59。

## 1. 出貨包是什麼

| 項目 | 內容 |
|---|---|
| 檔案 | `submission/cadc1013_0828b_final.tar.gz`(1.197 GB,34 個檔案) |
| md5 | `751f7e37e18c603262c15b04a7f4b066` |
| 演練結果(乾淨 Python 3.13 venv、官方 evaluator、本機 load 24–28) | **1.1009,100/100 feasible,平均 0.37 s/案,最長 1.21 s,第一案 0.052 s** |
| selfcheck | `cuda_available=True`、Flow 模型 step 300000 載入、`[legal-guard]` 0 次觸發、無 `[pool-fallback]` |
| 與 8/27 包的差異 | ① Flow prior 換成 **tail-tilt fine-tune 300k EMA**(同架構、同載入路徑,只換權重);② 新增 **最終合法性守門**(`PARTNER_FINAL_LEGAL_GUARD`,合法時輸出完全不變,非法時才換成驗證合法的備援,成本 0.3 ms) |
| 後備包 | `cadc1013_0828_final.tar.gz`(md5 `b25aaa7e…`,同上但無守門)、`cadc1013_0827_final.tar.gz`(md5 `163b8854…`,Flow v1) |

環境設定全部在包內的 `op_wrapper.py`(主辦 A20 確認會原樣使用),`requirements.txt` 已把 torch/numpy/scipy/numba/shapely/matplotlib/tqdm/requests/threadpoolctl 全部 pin 好;評測機 driver 580 / CUDA 13.0,我們釘的 `torch==2.6.0`(cu124 wheel)相容。

## 2. 上傳前要做的事(8/30–31,挑機器安靜的時候)

```bash
cd /ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning
bash scratchpad/gate_chains_0828/dryrun7.sh      # 重組包 → 新鮮解包 → 乾淨 venv → 官方 evaluator 100 案
```
看 log 最後幾行:`[selfcheck] cuda_available=True`、`loaded flow model step 300000`、`legal-guard fires: 0`、`feasible=100`,md5 應為 `751f7e37…`(同一工作樹重組會得到同一包)。只上傳 `cadc1013.tar.gz` 這一個檔。

分支 `final-sprint-0827`(最新 commit `79c74d2`)還沒 push(機器上沒有 GitHub 憑證);`submission/final-sprint-0827.bundle` 是完整的 git bundle,有憑證的人可以 `git bundle` 匯入後 push。

## 3. 今天(8/28)做了什麼、結論是什麼

**促轉(入包)**
- **Flow prior 換成 fine-tune 300k**:用 tail-tilt 抽樣(大 n 案抽多一點)從 v1 續訓 300k 步、cosine 退火、取 EMA。證據:official 7 組配對 −0.003~−0.006(且 rep 間變異只有 v1 的 1/3)、v3 −0.008、v5 −0.012、v6 持平,runtime +0~2%;慢 CPU 模擬(預算 ÷1.45)下 official −0.008、v3 持平 → 在比賽機速度下也不吃虧。
- **最終合法性守門**:壓力測試時發現一個 infeasible 輸出,追下去是合成資料產生器把兩個 preplaced 方塊撒成互相重疊(輸入本身不可能滿足),**solver 沒有 bug**;守門仍然加上當保險,因為 hidden 一案 infeasible 就是 ×8 的罰。

**撤回 / 否決(都量過,不要再開)**
- polish headroom 0.6 s 與 `FLOW_ANTITHETIC=0`:各自 4 組配對都「顯著」,但兩個疊在一起、同一條鏈直接對現包量,反而 +0.005、runtime +8%;分開再測也不進。**教訓:本機 ±0.01 的效應 4 組配對分不出真假,任何 knob 入包前必須「完整候選 env vs 現包 env」同鏈 ×4、兩種臂序**。
- `FLOW_SOLVER=heun`(+0.2,sampler 時間翻倍把模型臂關掉)、預算表 ×1.08(raw 小賺但 runtime +5–8%)、fine-tune 90k 快照(中段 n=76–89 崩塌)、按 n 路由雙 checkpoint(v3 +0.032)、第二輪 fine-tune(沒有 gate 時間)。
- refine ladder 的席位回收 / 自適應 rung-0:普查後判定可回收的空間 ≤0.002,遠低於噪音下限,不做。

**主辦 QA(8/27 版)重點**:決賽用 beta 同一組 hidden;op_wrapper.py 原樣使用;`__init__` 暖機不計時;hidden 連線用同一生成法、噪音只加 shape/placement(所以我們自製的 pin 偏移套 alpha_1 確定無關,已退場);requirements 非空就一定建 venv,所有依賴要自己 pin。

## 4. hidden 到底偏不偏離 public?

用 beta 的逐案結果和本機用「同一個 8/12 包、同樣 CPU-torch 失效狀態」重放 official 來比:
- hidden 的 block 數分佈與 public **逐 test_id 完全一樣**(四個 n 帶各 55/14/15/16 案、n≥105 佔 74% 權重),就是同一組 config 重新加噪。
- n<105 各帶 hidden 反而略易或持平;**只有 n≥105 帶 hidden 較難 +0.04/案**(HPWL 相同,差在 area 與 violations)。加權後 hidden ≈ public +0.02。
- 所以決賽 hidden raw 預期 ≈ 本機 +0.01~0.02(約 1.11–1.13)。beta 的 1.314 是部署失敗(CPU torch、requirements 空檔),不是資料。

另外對出貨包跑了 106 個極端合成案(內部 preplaced 障礙、密集 MIB/cluster、極端 fixed 比例、零/超多 pin):0 例外、runtime 有界(n=120 最長 1.33 s),solver 自己引入的重疊為 0。

## 5. 分數現況 vs 目標(誠實版)

| 套 | 現況(本機安靜時) | 目標 |
|---|---|---|
| public(official 100) | 1.10–1.105 | < 1.08 |
| shadow v3 | 1.13–1.15 | < 1.12 |
| shadow v5 | 1.10–1.11 | < 1.12 ✓ |
| shadow v6 | 1.12–1.14 | < 1.12 |

public 1.08 沒達到。剩下的 ≈0.02 幾乎全在「column 通道出貨」的案子(official 27/100,加權超額 +0.009):這些案模型候選經 refine 後輸給 column SA,是候選品質問題,不是排程或參數;所有 solver 端的 knob 家族已經全部量到噪音下限。三天內沒有可靠的槓桿,所以決定收手、以穩定出貨為優先。

## 6. 資料在哪

- 交接總覽:`docs/project-status/2026-08-28-final-sprint-handoff.md`(打包流程、方法鐵律、否決清單)
- 逐項證據:`docs/experiments/2026-08-21-post-beta-p0-execution.md` §17o–17af
- 官方 QA:`docs/official/C_QA_20260827.{pdf,txt}`
- gate 結果 JSON/log:`artifacts/shadow_gate_runs/{p3,n,ft,rt,fc,fd,po,pl300,em}*`;dry run:`artifacts/shadow_gate_runs/dryrun{6,7}_pack_off.json`
- 今天用的鏈與工具:`scratchpad/gate_chains_0828/`(含 `dryrun7.sh`、壓力測試、beta 重放);fine-tune 配方:`scratchpad/flow_ft_0828/STATUS.md`
- gate 工具:`scripts/gate/{run_gate5.sh,run_shadow.sh,analyze_pairs.py,band_pairs.py,ev_rt.py}`
