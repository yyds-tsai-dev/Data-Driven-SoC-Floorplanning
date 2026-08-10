# 2026-08-10/11 戰役日誌:frame-scale 促轉、IC/DC 引擎 gate 鏈、rung-(−1) 基建

**分支** `5.6-sol-reduce-time`;促轉 commit `fc2b354`。**Goal**:noRT 1.01 @ avg ≤0.3s(goal hook 錨定)。
**出貨點軌跡**:1.1647(c2dd42bb,0810)→ **~1.155-1.159 @ ~0.29s**(frame-scale 促轉後;repack v2 618755a7)。

## 1. 促轉:PARTNER_FRAME_SCALE_LADDER=1 + SET=1.02

- **機制**:direct 通道 rung-0 合法化框架寫死 `1.08×area_ref`(tid87 磨損解剖發現)——每個 direct 候選固定繳 +0.05~0.08 area_gap;golden 實測 packing 1.001-1.012×。單步常數位移 1.02。
- **證據**:7 對成對 full-100(3 對 frontier 括號 + 2 對確認鏈 + 2 對促轉錨定)6/7 同向,mean −0.0155 / median −0.0124;與 inert null 臂完全分離;機制三層驗證(離線重放 area_gap 隨 scale 1:1 回收、oracle 注入 tid85/87/91 1.08-1.13→1.01-1.05、真實 full-100);runtime avg/max 持平、100/100 feasible 全程;`[fs]` debug 確認觸發。
- **判死**:多階梯(1.02,1.05,1.08)離線 +0.0053 反劣(額外 rung-0 嘗試偷 refiner 時間)。分解:1.00 臂(rung-0 必失敗)已拿 −0.018 ⇒ 大半收益=「別用未退火的 1.08 框成交」;**後續正交槓桿:rung-0 成交後補 `_tighten`(expand 路徑都有、rung-0 成交路徑沒有)**。
- **repack v2**:`submission/cadc1013_0811_framescale.tar.gz` md5 `618755a7`;closure CLOSED 26/26、解壓 full-100 100/100 ×6、首案 0.054-0.076s 無冷 JIT、包內成對 ON/OFF 3/3 同向 −0.0055(量測窗被 GPU 訓練污染 +0.007,OFF 臂 1.1716 vs 0810 乾淨 1.1647 為證;校正後 ON ≈1.159)。舊包 c2dd42bb 未動,上傳擇一(deadline 8/12)。

## 2. IC/DC 引擎 gate 鏈(docs/design/2026-08-10-icdc-engine-bet-design.md)

- **G0-a 過**(死刑條款 q<1.05 不觸發):交叉點 q_cov=1.134;曲線嚴格單調;7 注入臂無一劣於 control。承重修正:覆蓋帶=n≥100(w=0.8264);封閉式 full-100=0.8052·q_cov+0.2571(中段殘差 ±0.0002);同質化稅實測 −0.0105(舊 0.035 高估)。
- **G0-b 過(四判準)+ 天花板改寫**:`PARTNER_LEGAL_ADMIT` rung-(−1) 合法即錄取(default off)。**完美 prior + 全覆蓋 = 1.011268 vs 資產 1.011256(磨損 +1.2e-5,98/100 bit-exact)⇒ 下游對硬合法 prior 零損耗**;舊 0.054「搬運磨損」= guard MIN_N=95 覆蓋洞(24/33)+ 小 n 採樣延遲吃光預算(6/33)+ rung-0 真失敗(3/33)的假象。全覆蓋成本 avg_rt +15%(小 n 採樣)。**KPI 1.01 = 兩個純問題:硬合法 prior + 小 n 便宜採樣**。
- **G1-a(進行中)**:TFDL 落地(pin 下界化+drift 梯度;**shapely grouping 浮點位元敏感** ⇒ bit-exact 壓實通道;DAG 凸組合 seating 一次 −0.97);energy pooled Spearman 0.9904 過 gate(有理式飽和鬆弛是關鍵);兩次發散根因=線性輔助項的病態膨脹最優解(已飽和化);**未訓練 control bank:合法率 40%(唯一失敗模式=preplaced drift,結構性)、官方 cost 中位 2.89**;規格線 q_cov≤1.0964;死刑條款(FT 後中位>1.5 無趨勢)待 v2 收斂判定。方向訊號:結構槓桿(seating −0.97)≫ 訓練(400 步 −0.28)。

## 3. 本波判死(併入 judged-dead 清單)

guard-MIN_N=0(+0.0019=旗標碰不到的帶的噪音;n=90-94 無軌跡可仲裁,rung-(−1) 才是正解)|retrieval 餵料(離線 0/200 + 解算器內 0/337 硬合法;transfer 換尺寸破壞 packing + MIB 同形未處理;R4 判死不需重審;0.3s 檔 retrieval 代價 +0.035)|DIRECT_WARM(deadline-bounded 下暖機無效反噬)|dm010(direct 中帶 4/4 敗給 column)|QSWEEP/DISC 吞吐轉換(6臂×3 括號:無訊號/3/3 反向;kernel 留 dormant)|SA_ADAPTIVE/RACING/PROXY_ALIGN|COL_NARROW(出貨組態未執行實錘,−0.0126=雜訊)|imitation 訓練(α 災難谷,α>0.8 才起飛)|learned-ranking 整類(G1=0.0005)|T2′ frame(oracle 反證)|R4 預算重分配。

## 4. 量測紀律沉澱(0807-0811)

單 rep 噪音 ±0.012、3-rep 兩臂差 2σ 地板 0.0078 ⇒ 促轉門檻 3-rep ≥0.008 或 ≥8 reps;每鏈帶 inert null 臂;跨 session 絕對值不可比(deadline SA 對機器負載敏感,OFF 錨定漂移 1.1647→1.1716 實證);live-tree eval 禁與同檔編輯並行;判讀一律加權 total_score_no_runtime;GPU 訓練與 CPU eval 同機互污(+0.007 級),headline 數字必須乾淨窗口。

## 5. 未決/在途

G1-a v2 訓練收斂與死刑條款判定|repack v2 乾淨窗口 headline 補釘(verify_full100.sh,GPU 空後)|DIRECT_SEAT_FIX(官方分 −0.0031)入 final repack 候選|rung-0 後補 _tighten(正交槓桿)|preplaced-aware 拓撲抽取(G1 合法率天花板 40% 的解)|beta 重傳上傳決策(8/12,新包 618755a7 vs 舊包 c2dd42bb)。
