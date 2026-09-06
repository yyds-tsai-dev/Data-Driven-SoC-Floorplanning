# 2026-08-31 決賽日交接(上傳已完成;deadline 23:00 UTC+8 = 15:00 UTC)

## 已上傳
- **`submission/FINAL_UPLOAD/cadc1013.tar.gz`,md5 `3f2cda42c88338fbb32d7670052fc895`**(B′:FT2 flow prior + FLOW_SLOTS16/NREF12 + legal guard 面積修復 + runtime 微優化 + mid 預算表)。
- 最終演練(dry run 12,乾淨 py3.13 venv、官方 evaluator):**1.0858、100/100、avg 0.363 s、max 1.31 s、首案 0.050 s、cuda_available=True、flow step 250000、legal-guard 0 次**。逐案 JSON `artifacts/shadow_gate_runs/dryrun12_final_off.json`。
- 隊友獨立複測:1.1174(他機、load 不同)、100/100、0 exception、兩顆模型載入——健康。
- QA 合規核對:A14/A20/A21/A22/A25/A27、48 核/128GB、34 檔、無絕對路徑、requirements 全 pin(scipy 在列)。舊包 11 顆在 `submission/old_packages/`。

## 若要在 15:00 UTC 前再換包(僅在有「過 gate 的更好候選」時)
1. 改動 → 四套 gate(`scripts/gate/run_gate5.sh`,同鏈 ×4 兩序;±0.01 效應 4 reps 分不出,不疊噪音級 knob)。
2. `bash scratchpad/rtaware/apply_pack_variant.sh ft2 s16 [k20]` → `bash scratchpad/rtaware/dryrun12.sh`(重組 → 新鮮解壓 → 乾淨 venv → 官方 100)。
3. 看:`feasible=100`、`cuda_available=True`、`step 250000`、`legal-guard fires: 0`;上傳 `pack_test12/cadc1013.tar.gz`。md5 會隨 tar mtime 變,內容驗證用解壓 diff。

## 現況分數(安靜機 ×4 平均)與差距診斷
- B′:public 1.087 / v3 1.111 / v5 1.086 / v6 1.100。對手三組 public <1.01、selfgen 1x 1.041–1.043(我們 1.076)。
- 差距 ≈0.07 已定位 = **模型 seed 品質**(§18l oracle:golden 餵我們的 ladder → 1.013;ladder 不 lossy)。violations 0.025(boundary locked 類為主)+ area 0.041 + hpwl 0.034。

## 已量死、勿再開(全部在 docs/experiments/2026-08-21-post-beta-p0-execution.md §18a–18u)
fine-tune round 3(x0/hpwl)與 round 4(bd/cg/mib 權重)、FLOW_SLOTS 20/24、oversample+prescreen、FLOW_STEPS 16、LS 解析種子、pin-frame 席位重測、CLUSTER_GLUE、VKILL、k25、grouping 修補預算(0/91 bit)、runtime 微優化以外的 runtime 工作(98% deadline-bound、M=1.0 已貼 floor)。k20(D′)不上:48 核專用機 M≈1.0,k20 只付 raw。POOL 維持 24(40 量過更差、48 無處驗證)。

## 賽後才有意義的方向(證據齊)
1. 預測器換架構:直接預測 `tree_sol`(離散、合法 by construction)或大幅提高 x0 精度——seed 到 golden 1% 內,我們的 ladder 就能出 1.013。
2. cluster loss 換連通性項(現行 `cluster_gap` 在 cluster 裂 15 塊時仍=0,§18q 審計)。
3. locked 牆線類:由 boundary tag 線定框的表示法(現行「先定框後貼」到不了)。

## 工具索引
gate:`scripts/gate/{run_gate5.sh,run_shadow.sh,analyze_pairs.py,band_pairs.py,ev_rt.py}`;runtime-aware:`scratchpad/rtaware/rt_pairs.py`;打包變體:`scratchpad/rtaware/apply_pack_variant.sh`;演練:`scratchpad/rtaware/dryrun12.sh`(B′)/`dryrun13.sh`(D′);探針:`scratchpad/rtaware/{golden_probe.py,vprobe.py,group_diag.py,bridge_budget_probe.py,ls_seed_probe.py}`;對手面板:`artifacts/shadow_hidden_suites/ext_selfgen/`、`ext_proxy/`。

## 08-31 續壓(使用者授權「已繳交、再試」)— 全部不促轉,B′ 定案
- R1 尾帶 floor-safe 加時(κ=0.232):四套 wash(off +0.0007/v3 −0.0026/v5 −0.0023/v6 −0.0003,CI 全含 0)。
- QSWEEP+DISC numba(deep-reasoner 找到的 §18u 漏軸):快篩 −0.0037 → 四套 v6 +0.0061 [+0.0000,+0.0121] 退,否決;DISC 單獨 v6 +0.0085 [+0.0010,+0.0173] 退,否決。
- deep-reasoner 兩輪反證:其餘 knob 死碼/重複/負 EV;M=1.0 時 runtime 軸全部可得空間 = 0.0004(§18a 校正);GPU 風險關閉(A100=sm_80 在 wheel arch list、seat_ts 門檻餘裕 2.76×)。
- 證據:experiments doc §18v;chain log `~/.claude/jobs/06af0e53/tmp/chain{R1,Q,QF,D}.log`。

## 10:31 UTC 更新:soup a50 過 gate + 演練 → C′ 候選包就緒,**等使用者裁定是否重上傳**
- chainSF 四套 ×4:off −0.0070 [−0.0149,+0.0003] / v3 −0.0010 / v5 −0.0010 / v6 +0.0045(CI 含 0);6 個 official 配對 rep 一致負向;runtime 持平。
- dryrun14(C′ = B′ + FLOW_CKPT=`flow_matching_ft0831_soupa50_250k_ema.pt`):**1.0834、100/100、cuda=True、step 250000、guard 0、首案 0.049s**。
- 候選包:`submission/cadc1013_0831_soupa50_final.tar.gz`(= pack_test12 同檔)。**線上仍是 B′(3f2cda42)= 保底**;要換就上傳候選包,不換不動作。
- 工作樹現為 C′ 設定;回 B′:`bash scratchpad/rtaware/apply_pack_variant.sh ft2 s16`。

## 15:01 UTC 收案
- deadline 已過。兩包完好:B′ `FINAL_UPLOAD/cadc1013.tar.gz`(3f2cda42,原線上)、C′ `FINAL_UPLOAD_C/cadc1013.tar.gz`(6d0ca94e,QA 驗畢:34 檔、0 pycache、唯一 diff = flow ckpt + op_wrapper 一行;14:47 UTC 給出上傳綠燈與建議)。
- **最終提交 = 使用者實際上傳者(待使用者確認:C′ 或 B′)。**
- C′ 證據鏈:official 6-rep −0.0066 [−0.0134,−0.0002]、v3/v5 wash、v6 +0.0059(CI 含 0);dryrun14 1.0834/100/100。
