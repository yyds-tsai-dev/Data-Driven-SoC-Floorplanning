# 2026-07-07 Gate 0 order 通道上限 probe(KILL)+ E2 tail 預算 probe(SURVIVES)

出處:深度研究 v3 合流報告([docs/research/2026-07-07-deep-research-v3-model-pipeline.md](../research/2026-07-07-deep-research-v3-model-pipeline.md))排序表 Gate 0 與 #6;判準皆預註冊於該報告。

## Gate 0:golden-oracle order 上限 probe — **order-only 通道 MEASURED KILL**

**假設與預註冊判準**:量「order 通道上限」——order 來自 golden 幾何抽取(= golden-like order 的定義),shapes/其餘輸入來自可實現 hint 來源。(a)+(b) 都 ≥1.228 → ML-order 全線降級;(b) 顯著 <1.2297 → Route C 上限確立。

**實作**(僅 [scripts/probes/gen_decoder_probe.py](../../scripts/probes/gen_decoder_probe.py),+46/−8,src 未動):`--order-hints {self,golden}`(default self 零行為變化)——`build_order_dags` 的 centroid 來源可換成 `_golden_rects`,**ws/hs、decode/exact-area/boundary-snap/polish 全部維持 --hints 來源**。語義註記:axis 選擇的 normalized separation 分母用的是 --hints 來源的 half-extents(「order 通道對照我們蓋得出來的 shapes」的部署語義)。Sanity:`--hints golden` self vs golden order **逐位一致**(1.048797517193381,3 案)→ 接線正確。

**結果(全量 100 案)**:

| 變體 | weighted no-rt | legal-fallback | 對照 |
|---|---|---|---|
| (a) production shapes × golden order | **9.2090** | 79/100 | 同 run paired baseline **1.2724**;decoder **0 勝 100 敗**;portfolio(min) deploy delta +0.0000 |
| (b) diffusion shapes × golden order | **9.3663** | 74/100 | self-order 同 ckpt 同日對照 **8.9496**(fallback 48/100)→ golden override **反而 +0.42** |
| (參照)golden shapes × golden order | 1.0488(3 案 sanity)/ 歷史 1.0535 | 0 | — |

非 fallback 案機制:boundary 違規 ~27–34/案、grouping ~13–21/案、tail area_gap +2.3–4.7,polish 全數 reject——不是修不好,是結構性不相容。

**判定:預註冊判準雙雙觸發 → order-only 通道 KILL(勿重議)。**

**機制(本 probe 的核心產出)**:0.18 headroom(1.23→1.05)需要 **(order, shape) 聯合構型**——golden 相對順序以 golden 長寬比為前提;把同一 order 用非 golden shapes 實現,DAG 實現爆 whitespace、牆/群結構全失,連 diffusion 自身一致的 order 都比 golden override 好(8.95 < 9.37)。**order 單獨攜帶的資訊 ≈ 0。**

**推論級聯**:
1. Route C v1「pair-cost head → order override」捷徑在**零訓練投資**下判死;LaMPlace 蒸餾 / Vlastelica blackbox diff 作為 order-only consumer 一併降級(除非未來出現同時供給 shapes 的 consumer)。
2. **ML 唯一存活路徑收斂回座標+形狀品質 = scale probe(ep5-6,~0709 判定)**;order 由幾何抽取對模型座標免費取得(98.7%),pair head 不需要。
3. golden→1.0535 上限仍成立(order+shapes 聯合 golden);研究 v3 的 promotion bar(area_gap<+0.3、fill>85%)正是 shape 品質 bar,方向再獲確認。
4. Gate-before-invest 的價值:整條訓練路線的生死在 ~25 分鐘、零訓練成本內判完。

## E2:tail 時間預算 ×2 — **SURVIVES kill bar(今日唯一)**

**假設與預註冊判準**:`_time_budget` 是純 n 函數(clamp 上限 24s);tail n≥116 五案佔加權 37% 且 N3 顯示 tail 吞吐是地板 → 放寬 tail 預算讓 SA 多收斂。kill bar:tail-5 配對加權改善 < 0.003 → kill。

**實作**([column_backbone.py:28-53](../../src/floorset_arch/legalizer/column_backbone.py)):`FLOORSET_TAIL_BUDGET_SCALE`(float,default 1.0)/`FLOORSET_TAIL_BUDGET_N`(int,default 116),clamp 後乘(可突破 24s 上限);default 逐位一致;env 解析失敗回退。測試 [tests/test_tail_budget.py](../../tests/test_tail_budget.py) 6 passed + SA 等價 30 passed 1 skipped。

**結果(相鄰配對 run,同機同負載;scale=2.0)**:

| 量 | control | scale2 | delta |
|---|---|---|---|
| total_score_no_runtime | 1.2279 | 1.2280 | +0.0001(全量持平) |
| **tail n≥116 配對加權** | — | — | **−0.00441,5/5 全勝** ✓(bar 0.003) |
| 未動的 95 案(噪音對照) | — | — | 54/42 對稱,無系統性漂移 |
| tail runtime | 18–23s | 36–45s | ×2 生效確認 |

tail per-case:d_cost 全負(−0.0006 ~ −0.0361,case 98 最大);**v_rel Δ=0.0000 全部**——增益純 HPWL/area(SA 收斂),機制與設計一致。全量持平是因 5 案增益被其餘大 case 的對稱噪音抵消;因果歸因以「唯一被改動的 5 案」配對讀數為準。

**第二組確認配對(同日稍晚,相鄰 run)**:tail 加權 **−0.00398,4/1 勝**(case 99 本組 +0.0099,單案 SA 噪音;其餘四案 −0.004~−0.029);全量 1.2271→1.2249(−0.0022);v_rel 仍全零。**兩組合併:10 個 tail 配對觀測 9/1 勝,效應 −0.0040~−0.0044,雙雙過 bar——效應確認。**

**判定:存活,score 證據已足(兩組獨立配對)。** 促轉剩餘前置:
1. **RTF 提交期風險**(CONTEXT.md「Official RuntimeFactor」):tail=最重案、慢懲罰不封頂;runtime ×2 → RTF 因子 ×2^0.3≈1.23,若我方 tail runtime 已 ≥ field median,runtime-included cost 的損失會吃掉 no-rt 增益。本地主指標是 no-runtime,此為 **submission 策略決策(user 層級)**——旗標維持休眠,列入提交前決策清單。
2. 可選延伸(未做):`FLOORSET_TAIL_BUDGET_N=100`(n≥100 帶 21 案)、scale=1.5 劑量反應(換取較低 RTF 曝險)。

## Repro

```bash
# Gate 0(a)/(b)(從 FloorSet/iccad2026contest,.env sourced,PYTHONPATH 含 src)
uv run python <repo>/scripts/probes/gen_decoder_probe.py --hints production --order-hints golden \
    --production-cache <sp>/prod_layouts_gate0.json --out <sp>/gate0_prod_goldenorder_100.json --verbose
uv run python <repo>/scripts/probes/gen_decoder_probe.py --hints diffusion --order-hints golden \
    --diffusion-checkpoint <repo>/checkpoints/diffusion_latest_0705_ns150000_ep10_diffhgt_lite_h256_l4_steps1000_acc16_bs4.pt \
    --diffusion-steps 32 --diffusion-samples 4 --out <sp>/gate0_diff_goldenorder_100.json --verbose
# (b) 對照:同指令去掉 --order-hints golden

# E2(注意 PATH 需含 ~/.local/bin)
bash scripts/eval_total.sh --output control.json
FLOORSET_TAIL_BUDGET_SCALE=2.0 bash scripts/eval_total.sh --output scale2.json
python3 <sp>/e2_analyze.py
```

工件:session scratchpad `gate0_{prod,diff}_goldenorder_100.json`、`gate0_diff_selforder_100.json`、`e2_{control,scale2}.json`、`e2_analyze.py`、`prod_layouts_gate0.json`(production layouts cache,retest 免重跑 SA)。
