# Retrieval R4 Gate：forced quota 退化，CP-SAT teacher 路徑停止

日期：2026-07-23

Branch：`codex/5.6-sol-partner-candidate-research`

結論：**R4 fail；Retrieval 維持 opt-in/off，CP-SAT golden-teacher 路徑不啟動。**

本報告評估的問題是：train-only、same-`N` Retrieval 是否能在不增加 Partner deadline 與 refine capacity 的條件下，以兩個 transferred candidates 取代 Direct candidates，改善 full-100 `total_score` 或 `total_score_no_runtime`。本題 cost 越低越好。

## 固定實驗契約

- Evaluator-facing solver：`partner/my_opt_claude.py`
- Direct checkpoint：step 1,139,000；control/treatment 使用同一份 checkpoint snapshot
- Retrieval index：`artifacts/retrieval/pilot64`
  - FloorSet training split only
  - 100 個 same-`N` shards，`N=21..120`
  - 每個 `N` 64 records，共 6,400 records
- 第一個 R4 treatment：最多兩個 Retrieval candidates，固定總候選/refine capacity；不增加 case deadline
- Control/treatment 共用：`PARTNER_DIRECT_MIN=2.5`、`PARTNER_PRESCREEN_V=1`、`PARTNER_NREF=15`、`PARTNER_OVERSAMPLE=4`、`PARTNER_TAG_ANCHOR_EXTRA=3`、`VKILL_OFF=1`
- Retrieval disabled path 已由 Task 5 review 證明維持原 Direct ranking/behavior；Retrieval index path 使用 repo absolute path

## Full-100 paired 結果

| 指標 | Direct control | forced Retrieval slots=2 | Δ treatment-control | 判讀 |
|---|---:|---:|---:|---|
| `total_score` | 2.0003461328 | 2.0131576739 | **+0.0128115411 (+0.640%)** | 退化 |
| `total_score_no_runtime` | 1.1257981382 | 1.1328798292 | **+0.0070816910 (+0.629%)** | 退化 |
| feasible | 100/100 | 100/100 | 0 | 相同 |
| avg runtime | 4.865590 s | 4.863647 s | -0.001943 s | noise |
| median runtime | 2.026418 s | 2.025926 s | -0.000491 s | noise |
| p90 runtime | 14.338653 s | 14.336268 s | -0.002386 s | noise |
| max runtime | 23.864081 s | 23.865575 s | +0.001494 s | noise |

以 `|Δ|<=1e-9` 定義 tie，no-runtime paired outcomes 為 34 wins / 44 losses / 22 ties。`N>=100` 為 9 / 8 / 4，沒有穩定的大案 winner band。

Weighted no-runtime regression 的主要來源：

| case | N | per-case Δ | weighted Δ contribution |
|---:|---:|---:|---:|
| 93 | 114 | +0.090921 | +0.0044103 |
| 88 | 109 | +0.065154 | +0.0020835 |
| 98 | 119 | +0.019460 | +0.0014319 |
| 95 | 116 | +0.013042 | +0.0007474 |

較大的改善只有 case 96（weighted -0.0017891）、87（-0.0004936）與 99（-0.0004539），不足抵銷損失，也不能形成可描述的穩定 feature band。

按 block-count band 分解，全局 weighted contribution 為：

- `N<60`：-0.0000406，權重貢獻可忽略；
- `60<=N<100`：+0.0007269，退化；
- `N>=100`：+0.0063954，主導整體退化。

因此 full-100 已否定「forced two-slot replacement」；runtime 幾乎相同，不能以速度收益補償品質退化。

## 為何需要 candidate-provenance trace

Production helper `_select_ranked_source_quota()` 的 `retrieval_quota=2` 是保留名額，不是上限：只要兩個 Retrieval candidates 存在，即使它們排在 shared rank 最後，仍會排擠兩個 Direct candidates。Direct backfill 只在 Retrieval 不足時發生。

Full evaluator JSON 只有 final floorplan/score，不能分辨：

- Retrieval 是否在 raw shared prescreen 真正勝出；
- forced quota 是否只是刪掉較好的 Direct；
- transferred geometry 經同樣 repair/refine 後是否仍有優勢；
- distance、match cost、confidence 或 `N` 是否能預先辨識 winner。

因此新增 read-only `scripts/probes/retrieval_trace.py`，不修改 production selection。它使用實際 `_sample_direct_raw_preds()`、`_sample_retrieval_preds()`、`_rank_portfolio()`、`_select_ranked_source_quota()` 與 `_worker_refine()`，同時計算：

1. production forced quota；
2. hypothetical maximum quota：同一 shared rank 的 `order[:K]`，Retrieval 最多兩個但沒有保留名額。

每個 policy union candidate 使用 candidate-stable seed 與相同 refine allowance。Official offline score 由既有 `gen_decoder_probe.score_case()` 計算；runtime 與 median 同設為 1，因此此處只比較 no-runtime-equivalent geometry/constraint cost。

這不是 full solver replay：它不包含 concurrent column restarts、shared absolute deadline、fusion、phase-B 與 vkill；所有結論只標為 candidate-path evidence。

## Raw / prescreen / repair-refine 證據

### 1 秒 stratified trace

Cases：0、10、49、79、88、93、96、98、99。

- maximum quota 只在 2/9 cases 選到 Retrieval，共 3 candidates（case 93 一個、case 96 兩個）。
- 其餘 Retrieval 全落在 shared-rank 尾端；maximum quota 回到 Direct candidates。
- 低/中 N 的 Retrieval refine cost 約 2.19–4.08；Direct winners 約 1.03–1.30。
- 1 秒不足以穩定完成高 N refiner，因此此 artifact 只用於 raw rank 與低/中 N 診斷，不用來否定 tail refined quality。

### 4 秒 tail trace

Cases：79、88、93、96、98、99；每案 production `K=15`。

- 99 個 forced/maximum union candidate treatments，只有 1 個 failure。
- 12 個 Retrieval candidates 中 11 個完成 refine。
- **Retrieval refined winner：0/6 cases。**
- maximum quota 相對 forced：2 better / 4 equal / 0 worse。
- 兩個改善完全來自保留額外 Direct：
  - case 79：1.278166 → 1.194552（Δ -0.083614）；
  - case 98：1.101485 → 1.093690（Δ -0.007795）。

Maximum quota 真正選入 Retrieval 的兩案，Retrieval 仍遠差於 Direct winner：

| case | shared rank | Retrieval refined cost | Direct winner | 結果 |
|---:|---:|---:|---:|---|
| 93 | 0 | 2.997201 | 1.130406 | Direct win |
| 96 | 0 / 1 | 2.793983 / 2.791072 | 1.019451 | Direct win |

這代表 shared cheap prescreen 對 Retrieval 有 false positives：即使 Retrieval 被排到全體候選第 0/1 名，經同樣 refine 後仍明顯較差。

在本次 sampled candidate-path trace 中，match metadata 沒有提供可驗證的 selector。Case 99 有本次最佳 observed distance/confidence 組合（distance 0.0687、confidence 0.1165），兩個 Retrieval refine cost 仍為 3.065407 / 2.157823，而 Direct winner 為 1.383217。因 sampled trace 沒有 Retrieval winner，現有資料無法建立或驗證以 `N`、retrieval distance、match cost、confidence、D4 transform 或 shared rank 為基礎的 winner gate；這不是對所有可能資料或 selector 的普遍不可能性證明。

Raw transferred layouts 的 official cost 均為 infeasible penalty 10；artifact 沒有 raw violation-type breakdown，因此不能把 infeasibility 單獨歸因於 overlap。Standardized repair/refine 可讓絕大多數 sampled candidates feasible，但沒有產生 Retrieval winner。這個 observed failure mode 與 transfer/raw geometry/repair 路徑未保留可用優勢一致，但本實驗沒有介入 source teacher，不能據此判定更好的 teacher 在所有 consumer 設計下都無效。

## Selector 與第二輪 full-100 決定

Maximum quota 比 forced quota 安全，因為它不強制刪除 shared-rank 較高的 Direct；但目前證據顯示它的收益只是撤銷 forced-quota 傷害，不是 Retrieval contribution。

不再跑 maximum-quota full-100，理由：

1. sampled stratified/tail candidate-path traces 的 Retrieval refined winner 為 0；
2. maximum policy 預期主要重現 Direct control，另付約 0.23–0.33 秒 tail matching generation cost；
3. 沒有預先定義且可重現的 cheap feature gate；
4. 再跑 full-100 只能驗證「不強制替換即可回到 baseline」，不能滿足任何 R4 promotion condition。

Maximum quota 可作為未來 opt-in defensive cleanup，但仍應 default off，且不得宣稱 Retrieval 已晉級。

## R4 與 CP-SAT 決定

R4 promotion 三個條件全部不成立：

1. fixed-deadline paired no-runtime 沒有改善，反而退化 +0.0070817；
2. aggregate 不 neutral，且沒有 feature-identifiable winner band；
3. sampled stratified/tail candidate-path traces 中，transferred geometry 沒有在 standardized repair/refine 後形成 Retrieval winner。

因此：

- Retrieval R4 = **fail**；
- Retrieval production default 維持 disabled；
- 不做第二次 blind forced-quota repeat；
- 不做 maximum-quota full-100；
- **不啟動 OR-Tools CP-SAT golden-solution / medoid / local-window teacher refinement。**

既定 CP-SAT teacher gate 的前提是：既有 training `fp_sol` 經同一 Retrieval consumer 已先證明有用，然後才判斷更好的 teacher 能否增加收益。現在 sampled trace 對既有 labels 產生 0 個 Retrieval refined winners，也沒有可驗證的 cheap selector，故未滿足啟動 CP-SAT teacher 實驗的先決條件；這是停止投資的 gate 決定，不是 improved teacher 的不可能性證明。若未來重新設計 correspondence/consumer，必須先重新通過 R4，才能重開 CP-SAT teacher 路徑。

這個結論不等於 CP-SAT 對所有 floorplanning 角色都無效；它只否定目前提案中的「先用 CP-SAT 改善 golden teachers，再透過此 Retrieval consumer 提供 hidden-case initial solution」路徑。Online CP-SAT 與獨立 local solver 仍屬另一項研究假設，未在本 gate 實作或評分。

## Commands、artifacts 與 hashes

Full-100：

```bash
bash scripts/probes/run_retrieval_gate.sh
```

- `artifacts/partner_eval/cont_retrieval_direct_control.json`
  - SHA-256 `243b039286e1ea302b9f71cafc684b72fa4051712b2ea898c995a19d413c2955`
- `artifacts/partner_eval/cont_retrieval_r4_slots2.json`
  - SHA-256 `a1de8be6a542de0bad530953751307d7bf1307ade1646121ab6600c2633075db`

Stratified trace：

```bash
PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 PARTNER_OVERSAMPLE=4 \
PARTNER_TAG_ANCHOR_EXTRA=3 VKILL_OFF=1 \
uv run python scripts/probes/retrieval_trace.py \
  --index "$PWD/artifacts/retrieval/pilot64" \
  --checkpoint "$PWD/partner/checkpoints/direct_v2_cont/eval_retrieval_direct_control.pt" \
  --case-ids 0,10,49,79,88,93,96,98,99 \
  --selection-k auto --refine-seconds 1.0 --seed 9001 --device cuda \
  --output artifacts/retrieval/trace_r4_stratified.json
```

- SHA-256 `dafbdde73d8e6a1358f2c847c48031eb3d254c036eac77486e3784f1901569ac`

Tail trace：

```bash
PARTNER_PRESCREEN_V=1 PARTNER_NREF=15 PARTNER_OVERSAMPLE=4 \
PARTNER_TAG_ANCHOR_EXTRA=3 VKILL_OFF=1 \
uv run python scripts/probes/retrieval_trace.py \
  --index "$PWD/artifacts/retrieval/pilot64" \
  --checkpoint "$PWD/partner/checkpoints/direct_v2_cont/eval_retrieval_direct_control.pt" \
  --case-ids 79,88,93,96,98,99 \
  --selection-k auto --refine-seconds 4.0 --seed 9001 --device cuda \
  --output artifacts/retrieval/trace_r4_tail_refine4.json
```

- SHA-256 `e286048367b81d4a06ca9e829527c4708b4a7b041fee72bb2f001aace717272c`

## Relevant commits

- `8a72481` — opt-in fixed-capacity Retrieval candidates
- `4ce8b96` — two-slot hard cap and absolute gate index path
- `fd16bc5` — candidate-provenance trace
- `56b489a` — evaluator-faithful anchors/live pool capacity
- `2f6f3b1` — exact production pool-capacity parity
- `9a69398` — official offline scoring helper
