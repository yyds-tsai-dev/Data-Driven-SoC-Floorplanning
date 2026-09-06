# Partner 候選供給研究設計：Retrieval、CP-SAT 與 Flow Matching

## 1. 研究目標

本研究以 `partner/` 目前最佳管線為主體，以 `src/` 正式架構作對照，回答三個問題：

1. 現有執行速度與分數瓶頸究竟在候選生成、候選修復，還是 column simulated annealing（SA）？
2. 能否利用 FloorSet training solutions 建立 train-only case retrieval，為 unseen / hidden case 提供可保留品質的完整初始 layout？
3. 在 retrieval 通過後，CP-SAT 是否適合離線精煉少量代表解；Flow Matching 是否能以較少模型 forward 次數取代 Direct-v2 的 50-step DDIM sampling？

研究成功不是「某個新模組可以執行」，而是至少達成以下一項，且不得以額外 per-case wall time 換取：

- 降低全 100 案 no-runtime score，方向朝 `<= 1.08` 前進；
- no-runtime score 持平，但相對官方 alpha per-case median 的 runtime factor 明顯改善；
- 在固定 partner deadline、固定 refine slots 下，新增候選來源對一組可描述的 cases 形成可重現互補。

正式促轉仍以全 100 案 paired evidence 為準，不以 training loss、單案最好結果或 validation 記憶結果代替。

## 2. 已知基線與瓶頸

### 2.1 主管線與對照組

`src/solver/my_opt_claude.py` 是目前研究主管線：

```text
heuristic / legacy diffusion seed
  -> parallel column restarts
  + Direct-v2 batched 50-step DDIM candidates
  -> prescreen
  -> selected full-budget refinements
  -> final proxy selection
  -> optional violation kill
```

目前最佳全量證據約為：

- no-runtime score：`1.1162`
- average runtime：`4.87 s`
- maximum runtime：`23.70 s`
- feasible cases：`100 / 100`

主要證據位於 `artifacts/partner_eval/full100_t14_directmin.json`；較早的時間中性供給放大實驗見 `docs/experiments/2026-07-12-partner-supply-amplification.md`。

`src/architecture_v5_optimizer.py` 與 `src/architecture_v11_optimizer.py` 目前都走 `ArchitectureV11Optimizer`；啟用 column backbone 時，正式路徑的核心仍是 column-slicing、parallel-restart SA 與 refiner。`src/` 只作對照，不作本研究第一輪的主要修改面。

### 2.2 已量測的速度瓶頸

SA throughput probe 顯示：

| Block count | Approx. moves/s | Cache hit | 主要現象 |
|---:|---:|---:|---|
| 21 | 3631 | 82.2% | layout rebuild 已佔主要成本 |
| 68 | 3386 | 79.4% | geometry evaluation 持續主導 |
| 115 | 1385 | 64.4% | throughput 約降至小案四成 |

熱點集中在 `_layout_fast`、`_stack_column`、`_solve_column`。主要成本是每次 SA move 重新建立 column geometry；單純優化 HPWL 或 violation arithmetic 不足以解除大案瓶頸。

另一方面，`partner/` 實驗已顯示：在相同 wall time 下增加並改善 Direct-v2 候選供給，比調整 SA penalty、增加局部後處理或混合更多 column 解族更有效。這使本研究優先處理「候選供給品質與成本」，而不是先重寫整個 SA。

### 2.3 官方 runtime 參考

`docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv` 的 100 案中位 runtime 統計為：

- minimum：`1.612 s`
- median：`4.3655 s`
- mean：`4.64078 s`
- maximum：`11.828 s`
- sum：`464.078 s`

由於 alpha 是 hidden instances，本地 case id 與 block-count 對齊只能用於近似投影，不能宣稱是正式 alpha score。所有報告必須同時列出：

1. 本地 no-runtime score；
2. 實際 per-case wall time；
3. 以 alpha median 做的近似 runtime-aware projection；
4. 「hidden instance mismatch」警語。

本次 alpha submission 的程式版本推定來自 `v10-budget-proxy-runtime-grouping` 在 2026-06-18 或 2026-06-24 附近的 commit，但目前沒有足夠 artifact 可以唯一確認 exact commit。後續報告只能寫成「likely provenance」，不可把推定當成已驗證版本。

## 3. 已排除或降級的方向

### 3.1 Validation lookup 只作 oracle，不得促轉

FloorSet training loader 已提供 `tree_sol`、`fp_sol`、`metrics_sol`；evaluation adapter 也能讀取 polygon `fp_sol`。然而 validation / evaluation cases 不得加入正式 retrieval library。它們只可用於：

- 檢查 transfer upper bound；
- 診斷 training、sampling、repair pipeline 是否能保留好解；
- 形成清楚標記為 leaking 的 oracle。

正式 retrieval index 必須是 train-only。這也避免落入官方題目對 reverse-engineering generator / memorization 的合規風險。

### 3.2 單獨搬運 order 或 seed 不再作主線

既有實驗顯示：

- golden coordinates 作 column-backbone seed，改善約 `-0.001`，接近抽樣噪音；
- golden order 單獨和其他 shape / geometry channel 混接可能嚴重退化；
- order、shape、coordinates 必須形成 coherent layout。

因此 retrieval 不得只輸出 block order 給 `src/` column SA。它必須輸出完整 normalized `(x, y, w, h)` candidate，經 block matching、repair 與 partner direct-refine 路徑測試。

### 3.3 不先用 CP-SAT 重解全部一百萬筆

FloorSet-Lite 已有約一百萬筆 solution labels。即使每案 CP-SAT 只花一秒，serial compute 也約為 11.6 天；五秒約 57.9 天，六十秒約 694 天。大案還可能在時限內無法證明 optimal。

CP-SAT 只接受整數變數；連續座標、shape、pin offset 與 area tolerance 必須量化或離散化。因此時限內產生的結果應稱為 `CP-SAT-improved teacher`，不能在沒有 optimal certificate 時稱為真正 golden solution。

正式順序是：先用既有 `fp_sol` 證明 retrieval consumer 有用，再決定 CP-SAT 是否值得改善 teacher。

## 4. 整體架構

研究分成兩條可以獨立停止的分支，最後才作固定預算組合：

```text
Branch R: train fp_sol
  -> instance embedding / clustering
  -> top-K retrieval
  -> block correspondence
  -> coherent layout transfer
  -> repair / partner prescreen / refine
  -> [若通過] CP-SAT medoid 或 local-window teacher refinement

Branch F: Direct-v2 control
  -> conditional Flow Matching with same representation/conditioner
  -> low-NFE sampling sweep
  -> partner-compatible candidates
  -> fixed-budget prescreen / refine

Final portfolio:
  Direct-v2 + Retrieval + Flow Matching
  under the same deadline, batch cap, prescreen count and refine slots
```

Branch R 和 Branch F 必須先分開做 ablation。若同時換 teacher、sampler、candidate count 和 refine allocation，任何改善都無法歸因。

## 5. Branch R：Retrieval 的分階段證偽

### R0：資料與 leakage contract

建立明確的 dataset roles：

| Dataset | 可訓練 embedding | 可進正式 retrieval index | 可評估 | 可作 oracle |
|---|---:|---:|---:|---:|
| FloorSet training | 是 | 是 | 是 | 是 |
| Local validation / evaluation | 否 | 否 | 是 | 是，但須標 leaking |
| Official hidden | 否 | 否 | 最終評估 | 否 |

Index artifact 必須記錄 source split、case identifier、feature schema version 與 solution-label checksum，載入時拒絕未知或 evaluation source。

### R1：先證明「找得到相似 case」

第一版先限制為相同 block count，避免同時解決 variable-size matching。Instance embedding 必須與 block permutation 無關，初始 features 包含：

- block count、總面積、area distribution；
- aspect / fixed / preplaced 統計；
- net count、net degree、block degree distribution；
- block-to-block 與 pin-to-block connectivity 統計；
- boundary、cluster、MIB constraint 統計；
- normalized preplaced geometry；
- 可選的 graph encoder pooled embedding。

先比較三種簡單方法：

1. hand-crafted standardized features + cosine / L2；
2. hand-crafted features 分桶後的 cluster medoids；
3. 現有 graph conditioner 的 frozen pooled features。

本階段只檢查 retrieval consistency 與 oracle correlation，不接 solver。若 feature 距離和「source solution 可轉移後的品質」完全無關，停止複雜 ANN 或 CP-SAT 投資。

### R2：Block correspondence

對 retrieved source 與 target 建立 node matching cost：

- normalized area 與 shape feasibility；
- fixed / preplaced / boundary / MIB / cluster compatibility；
- node degree與 incident-net statistics；
- graph neighborhood embedding；
- preplaced anchors 的相對位置。

第一版使用 Hungarian assignment，且只處理相同 `n`。Hard-incompatible pairs 給予禁止或極大 cost。Matching 必須輸出 diagnostics：總 cost、最差 pair、hard-compatibility failures 與 confidence margin。

若 matching confidence 過低，retrieval candidate 應直接棄權，不得強行 repair。

### R3：Coherent layout transfer

Source layout 先以總面積尺度正規化，再依 target area / shape limits 重建每個 block 的 `(w, h)`；coordinates、shape 與 ordering 必須一起轉移。

需要測試的 transformation：

- uniform scale；
- x / y mirror；
- transpose（約束允許時）；
- anchor-aligned affine shift；
- target-area-preserving shape projection。

每個 retrieved case 最多形成小量 D4-equivalent candidates。所有 candidates 通過相同的 cheap overlap、HPWL、boundary / grouping prescreen，只讓極少數進入昂貴 repair/refine。

### R4：Partner 接入

Retrieval candidate 必須接到與 Direct-v2 等價的 candidate boundary：

```text
raw coherent rectangles
  -> cheap prescreen
  -> selected refine worker
  -> final picker
```

它不增加 case deadline；第一版應取代一個較弱的 column restart 或 Direct oversample/refine slot。必須報告：

- retrieval / matching / transfer latency；
- raw、repaired、refined 三階段品質；
- candidate 被 prescreen 保留率；
- final winner rate；
- 對 Direct-v2 的 case-level complementarity。

### R5：停止與晉級條件

Retrieval 只有在以下任一條件成立時晉級 CP-SAT：

- 固定總時間下，全量 paired no-runtime score 有穩定改善；
- aggregate 持平，但在可描述 case band 有穩定 winner set，且能用 instance features gate；
- candidate 在 repair 後仍保留 source layout 的 HPWL / area 優勢。

以下任一現象代表停止：

- retrieved candidate 大多在 matching 後失去相似性；
- repair 把 retrieved geometry 完全洗掉；
- candidate 很少通過 prescreen，或 final winner rate 接近零；
- 改善只能由 validation self-retrieval / label leakage 得到。

## 6. Branch C：CP-SAT 的正確角色

### C0：用途

CP-SAT 不作線上 hidden-case solver，也不作全部 training cases 的重解器。它只在 Branch R 通過後承擔兩種離線用途：

1. 精煉 cluster medoids、dirty labels 或 retrieval 高頻代表；
2. 固定大部分 layout，只重排局部 window / critical-net neighborhood。

### C1：有限 shape catalog

每個 movable block 依 area target、`±1%` area tolerance 與 aspect limits 預先產生少量整數 `(w, h)` candidates。用 selector literals 決定 shape，再建立 x/y interval variables 與 `NoOverlap2D`。

HPWL 以每條 net 的 `min_x/max_x/min_y/max_y` 輔助變數表達；pin offset 隨 shape 的變化以 table / element constraints 處理。Soft constraints採加權 penalty或 lexicographic objective，不直接假設與官方非線性 score完全等價。

### C2：兩種 probe

#### Small-case global probe

- 僅選 n 較小的代表案；
- 逐步增加 time limit；
- 記錄 feasible / optimal / gap / unknown；
- 與既有 `fp_sol`、partner result 比較。

目的不是立即上線，而是判斷 CP-SAT teacher 是否真的能勝過既有 labels。

#### Local-window probe

- 從 partner 或 retrieved layout 出發；
- 固定大部分 blocks；
- 只放開 critical nets、overlap region 或高 violation neighborhood；
- 加 displacement trust region，防止整體結構崩解。

這是預期較可行的主路，因為它縮小組合空間並保留 coherent global layout。

### C3：晉級條件

只有當 CP-SAT-improved teacher 相對原 `fp_sol` 在同一 retrieval / transfer consumer 下形成可測改善，才擴大 medoid coverage。若 teacher 指標更好但 transfer 後無差異，停止擴大；此時瓶頸仍在 matching / repair，而不是 teacher。

## 7. Branch F：Flow Matching 的分階段證偽

### F0：替換目標

第一優先替換 `src/solver/direct_model_claude.py::sample_direct()` 的 50-step DDIM candidate generator，而不是先重寫 legacy 4-step graph diffusion seed。理由是 Direct-v2 是目前經驗上最強的候選來源，而且 50 neural-function evaluations（NFE）留下實際降本空間。

保留以下 contract：

- 相同 graph / node conditioning；
- 相同 normalized layout representation；
- 相同 hard anchor channels；
- 相同 D4 augmentation；
- 相同 area-to-rectangle concretization；
- 相同 geometry、HPWL 與 constraint auxiliary losses；
- 輸出仍是 partner 可消費的 rectangle batch。

### F1：Flow objective

第一版使用 conditional rectified path：

```text
epsilon ~ N(0, I)
z_t = (1 - t) * epsilon + t * z_0
v_target = z_0 - epsilon
```

模型輸出 velocity `v_theta(z_t, t, condition)`。Sampling 從 `t=0` noise 積分到 `t=1` layout。Fixed / preplaced known channels在每個 integration step重新投影，避免只在最後修正 hard anchors。

### F2：最小正確性 probe

在小型 training-only subset 上 overfit，先證明：

- loss 能下降且 samples 接近 memorized layouts；
- integration 方向正確；
- padding mask與 variable block count 正確；
- hard anchors不漂移；
- Euler與 Heun sampler都能輸出有限、可 concretize 的 layout；
- checkpoint可由獨立 inference process 載入。

此 gate 失敗就不進行長訓練。

### F3：公平模型比較

Direct-v2 與 Flow Matching 使用相同：

- dataset split與 sample order；
- model capacity與 conditioner；
- augmentation；
- optimizer class；
- 一組固定累積 training samples 的 quality comparison；
- 一組固定 training wall-clock 的 compute-efficiency comparison；
- candidate batch size與 random-seed protocol。

Sampling matrix：

| Model | Solver | Reported budget |
|---|---|---|
| Direct-v2 | DDIM | 50 NFE baseline |
| Flow Matching | Euler | 4 / 8 / 16 / 25 / 50 NFE |
| Flow Matching | Heun | 4 / 8 / 16 integration steps，另報實際 NFE |

Heun 每 integration step 通常需要兩次模型評估，因此不得只以「steps」宣稱加速。

先作 candidate-only evaluation，不跑完整 SA：

- sampling latency、peak GPU memory；
- overlap fraction；
- boundary / group / MIB proxy；
- raw HPWL proxy；
- best-of-K與 candidate diversity；
- cheap repair 前後的 structure retention。

F3 通過條件至少符合一項：

- Flow Matching 在 `<=16 NFE` 匹配 DDIM-50 的 candidate quality；
- 相同 NFE 下 Flow Matching 明顯改善 quality；
- Flow Matching 提供可被 refine 利用、且與 Direct-v2互補的 winner cases。

### F4：Partner 固定預算接入

比較三組：

1. Direct-v2 only；
2. Flow Matching only；
3. Direct-v2 + Flow Matching，平分固定 candidate / refine capacity。

每組固定：

- per-case deadline；
- GPU batch cap；
- prescreen保留數；
- refine worker數與 slots；
- final picker；
- evaluator與 random seeds。

促轉條件：paired no-runtime改善可信，或 no-runtime持平但 sampling / total runtime顯著降低；alpha-median projection不得惡化。

## 8. 最終候選 Portfolio

只有獨立分支通過後，才比較：

| Ablation | Candidate sources |
|---|---|
| D | Direct-v2 |
| D+R | Direct-v2 + Retrieval |
| D+F | Direct-v2 + Flow Matching |
| D+R+F | Direct-v2 + Retrieval + Flow Matching |

所有組別共享完全相同的 case deadline和 refine capacity。分配策略先固定，再研究 feature-gated allocation。若新增來源只搶走更好的 Direct candidates，應停止組合，而不是增加 slots。

進階的 `retrieved layout -> residual Flow Matching -> target layout` 僅在 R 與 F 都獨立成功後研究。它需要可靠 block matching和成對 training targets，不屬於第一輪。

## 9. 評估方法

### 9.1 Case-level logging

每案至少記錄：

- block count與 constraint statistics；
- candidate source、rank與 prescreen score；
- generation、matching、repair、refine時間；
- raw / repaired / final HPWL、area、soft violations；
- final winner source；
- deadline overrun與 fallback；
- reproducibility seed和 artifact versions。

### 9.2 Aggregate metrics

每次實驗報告：

- feasible count；
- official-formula no-runtime score；
- average、median、p90、maximum與sum runtime；
- per-case delta與按 `exp(n/12)` 加權 delta；
- alpha median近似 runtime-aware projection；
- source winner count及其加權貢獻；
- 至少兩次 stochastic repeat，或 paired bootstrap confidence interval。

### 9.3 Runtime 計時邊界

離線 training、index construction與 CP-SAT teacher generation不計 submission runtime，但必須記錄 compute cost與 artifact size。線上必須計入：

- index load / warmup policy；
- case embedding與nearest-neighbor query；
- block matching與transform；
- Flow Matching sampling；
- prescreen、repair和refine。

不得把第一次模型或 index load藏在未被官方允許的初始化階段；需要分別報 cold-start與steady-state行為。

## 10. 實驗順序與停止規則

```text
Gate R1: existing fp_sol retrieval correlation
  fail -> stop R and C
  pass -> R2/R3 coherent transfer

Gate R4: partner fixed-budget utility
  fail -> stop C; diagnose matching/repair only
  pass -> CP-SAT small/global and local-window probes

Gate F2: Flow Matching overfit/sampler correctness
  fail -> stop F
  pass -> equal-training candidate-only comparison

Gate F3: low-NFE candidate quality or complementarity
  fail -> retain Direct-v2
  pass -> partner fixed-budget F4

Only passed R/F branches enter final D+R+F ablation.
```

每個 gate 都要允許「證偽並停止」。不得因已花訓練時間而降低晉級門檻。

## 11. 預期可行度

| 方法 | 可行度 | 目前判斷 |
|---|---:|---|
| 重解全部 1M training cases | 1/5 | 已有 labels，成本過高 |
| Train-only retrieval index | 3/5 | 值得先做，主要風險是 matching |
| Retrieved order 餵 column SA | 1/5 | 已有反證，資訊易被洗掉 |
| Coherent retrieved candidate 餵 partner refine | 3/5 | 有機會，必須實測 |
| CP-SAT 精煉 medoids | 3/5 | 只在 retrieval通過後 |
| CP-SAT local-window refinement | 4/5 | 比全域求解更符合工具優勢 |
| Flow Matching 直接全面取代 diffusion | 2/5 | 不保證少步仍保品質 |
| Low-NFE Flow Matching 作候選來源 | 3/5 | Direct 目前50步，值得公平測試 |

## 12. 交付物

研究完成時應留下：

1. train-only retrieval schema、index manifest與 leakage tests；
2. retrieval / block matching / transfer probe；
3. partner candidate-source adapter與 time-neutral ablations；
4. CP-SAT small/global與 local-window probe報告；
5. Flow Matching training/sampling smoke、candidate-only和 end-to-end報告；
6. 全 100 案 JSON evidence、per-case CSV與兩次以上 repeat；
7. 一份 promotion / rejection decision，明列被證偽路線。

## 13. 外部與本地依據

- Official Problem C：`docs/official/C_20260522.pdf`
- Official Q&A：`docs/official/C_QA_20260618.pdf`
- Alpha runtime medians：`docs/official/alpha_test/C_Median Runtime per Testcase(Alpha).csv`
- FloorSet paper：<https://arxiv.org/abs/2405.05480>
- FloorSet repository：<https://github.com/IntelLabs/FloorSet>
- OR-Tools CP-SAT：<https://developers.google.com/optimization/cp/cp_solver>
- OR-Tools scheduling / no-overlap examples：<https://developers.google.com/optimization/scheduling/job_shop>
- Flow Matching for Generative Modeling：<https://arxiv.org/abs/2210.02747>
- Flow Straight and Fast / Rectified Flow：<https://arxiv.org/abs/2209.03003>
- Column backbone seed evidence：`docs/experiments/2026-07-05-column-backbone-pivot.md`
- Model-channel evidence：`docs/research/2026-07-07-deep-research-v3-model-pipeline.md`
- Candidate-supply evidence：`docs/experiments/2026-07-12-partner-supply-amplification.md`

## 14. 已核准決策

1. 以 `partner/` 最佳管線為主，`src/` 作對照。
2. 以 no-runtime `<=1.08` 為品質方向，但同時使用 alpha per-case median檢查 total-score runtime風險。
3. Validation memorization只作 oracle，不得進正式 retrieval library。
4. Retrieval先使用既有 training `fp_sol`；通過後才用 CP-SAT精煉 medoids / local windows。
5. Retrieval傳遞完整 coherent layout，不只傳 order或單一 channel。
6. Flow Matching獨立測試，第一優先對照 Direct-v2的50-step DDIM。
7. Retrieval、CP-SAT、Flow Matching最後必須在固定 deadline與 refine capacity下做消融。
