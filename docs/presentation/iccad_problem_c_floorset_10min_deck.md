# Data-Driven SoC Floorplanning for ICCAD 2026 FloorSet Problem C

10-minute Physical Design Automation final project presentation.

## Presentation Design Decisions

| Decision          | Recommended answer                                                                                                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Audience          | PDA course audience familiar with floorplanning, not necessarily this repo.                                                                                                                    |
| Main claim        | The project combines learned geometric priors with a deterministic, legality-preserving floorplanning solver and evidence-gated v10 scoring decisions.                                         |
| Scope             | Present Architecture v5 production inference, training/checkpoint flow, and v10 policy. Do not present HGT as the production winner unless evaluator evidence promotes it.                     |
| Evidence standard | Use full-validation `total_score_no_runtime` as the main architecture and checkpoint comparison metric; use raw runtime, P90, max runtime, and runtime-aware total as gating/sanity signals. |
| Timing            | 11 main slides, about 40-55 seconds per slide, plus backup appendix slides for Q&A.                                                                                                            |

## Terminology Ledger

| Canonical term                  | First-use definition                                                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| FloorSet Problem C              | ICCAD 2026 contest task for data-driven SoC floorplanning.                                                                                 |
| Production Solver Path          | The single `solve()` path used by the evaluator.                                                                                         |
| Anchor-GNN Guidance             | Learned block-level geometric priors produced by a checkpoint and consumed by the decoder.                                                 |
| Selectable Anchor-GNN Encoder   | MPNN, Graph Transformer, or Local HGT encoder variants that share the same output heads.                                                   |
| AnchorGuidance                  | Runtime guidance object containing `rect_priors`, `priority`, `log_aspect`, and `pairwise_axis`.                                   |
| Hard Legality Gate              | Candidate ranking boundary that prioritizes missing blocks, overlaps, area, fixed-shape, and preplaced legality before soft quality.       |
| V10 No-Runtime Proxy            | Solver-internal candidate acceptance proxy for v10 no-runtime quality.                                                                     |
| V10 Evidence-Gated Budget Layer | Shared decision surface for allocating extra candidate/repair/portfolio effort only when validation evidence and instance risk justify it. |
| No-Runtime Quality Score        | Local full-validation metric equal to official quality and soft-violation factors with runtime adjustment fixed to 1.0.                    |

## One-Sentence Argument

In ICCAD 2026 FloorSet Problem C, we build an Architecture v5 solver that uses Anchor-GNN guidance as a geometric prior, keeps legality and soft-constraint handling in deterministic decoder/repair modules, and promotes changes through full-validation v10 no-runtime evidence with runtime-aware gating.

---

## Slide 1 - Title

**Data-Driven SoC Floorplanning for ICCAD 2026 FloorSet Problem C**

Physical Design Automation Final Project

Architecture v5: learned guidance, deterministic repair, and v10 evidence-gated ranking

**Chinese presenter note**

開場先說這不是單純「用 GNN 直接吐 floorplan」的專案。核心是 learned prior + deterministic physical-design solver。GNN 只提供幾何先驗；真正確保合法性、修 soft constraints、選 candidate 的，是後面的 decoder、repair、ranking layer。

---

## Slide 2 - Problem and Scoring Pressure

**Problem**

- Input: blocks, target areas, block-to-block nets, pin-to-block nets, pins, and constraints.
- Output: one `(x, y, width, height)` rectangle for every block.
- Hard constraints: no overlap, soft-block area within 1%, fixed-shape dimensions, and preplaced locations.
- Soft constraints: boundary placement, grouping connectivity, and MIB shape consistency.
- Objective: reduce HPWL gap, bounding-box area gap, soft-violation ratio, and runtime factor.

**Why v10 is hard**

- Infeasible cases are capped at a cost of 10.
- Case weights grow approximately as `exp(n/12)`.
- Large or constraint-dense cases can dominate the total score.

**Chinese presenter note**

這張講 Problem C 的輸入輸出。重點要把 hard constraints 和 soft constraints 分開：hard fail 直接進 infeasible penalty；soft constraints 不會讓 case infeasible，但會進指數懲罰。v10 最大改變是大型 case 權重很高，但不是只看最大 block count，constraint density 和 net density 也會讓中大型 case 變高風險。

---

## Slide 3 - Core Design Idea

**Learned priors should guide placement, not replace the solver**

- The Anchor-GNN predicts geometric priors, priority, aspect ratio, and pairwise orientation.
- The Production Solver Path remains deterministic and legality-aware.
- Encoder experiments do not change the decoder, repair, or ranking contract.
- Architecture changes are promoted by full-validation `total_score_no_runtime`.
- Runtime-aware total and raw runtime are used as submission safety checks.

**Design principle**

```text
ML predicts useful priors
      +
Physical-design logic enforces legality
      +
Evaluator evidence decides what becomes default
```

**Chinese presenter note**

這張是整份報告的 thesis。可以說：「我們沒有把 floorplanning 變成純 supervised regression，因為 contest output 必須 hard legal。模型負責提供 prior，solver 負責 legality 和 constraint-aware search，最後 evaluator evidence 決定哪些改動能變 default。」

---

## Slide 4 - Architecture Evolution: What Changed and Why

**From branches to the current design**

| Stage                            | Main attempt                                                         | What we learned                                                                          | Design consequence                                                              |
| -------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `arch-v1` / early `arch_new` | Build initial GNN-guided floorplanning path.                         | Direct learned geometry still needed strong deterministic repair.                        | Keep learned guidance as a prior, not as the whole solver.                      |
| `arch-v2` / `arch-v3`        | Add relative-order decoding, pairwise heads, checkpoint experiments. | Better training loss did not always mean better evaluator score.                         | Promote checkpoints by evaluator evidence, not validation loss alone.           |
| `arch-v4`                      | Add no-runtime scoring and runtime-aware diagnostics.                | Local runtime-aware score could over-penalize large cases due to local median artifacts. | Use `total_score_no_runtime` for architecture tuning; keep runtime as a gate. |
| `v5-graph-transformer-encoder` | Add Graph Transformer checkpoint and richer constraint context.      | Factor-derived context improved the current default checkpoint path.                     | Make encoder selectable while preserving the same downstream contract.          |
| v10 branch                       | Add v10 proxy, conditional budget, and narrow grouping bias.         | Soft-first acceptance and broad knobs could regress HPWL/area or runtime.                | Move promotion decisions into v10 proxy and evidence-gated budget layers.       |

**Chinese presenter note**

這張補充「為什麼現在長這樣」。講法是：早期 branch 證明 GNN guidance 有用，但不能取代 physical-design solver；中期 training/checkpoint 證明 validation loss 不能直接當 promotion 指標；v4 解決 local runtime-aware score 的偏差；v5 把 encoder 變可替換；v10 branch 則發現真正要改的是 acceptance/ranking/budget，而不是先再改 GNN 或 decoder。

---

## Slide 5 - Architecture v5 Production Flow

**End-to-end inference**

```text
Evaluator tensors
  -> parser.parse_instance()
  -> Instance
  -> _try_anchor_guidance()
  -> AnchorGuidance
  -> _candidate_specs() + budget_layer
  -> relative_order / optional beam candidates
  -> repair_placement()
  -> placement_metrics()
  -> v10_proxy_rank()
  -> best Placement
  -> list[(x, y, w, h)]
```

**Key implementation boundary**

- `src/architecture_v5_optimizer.py` is the evaluator-facing wrapper.
- `ArchitectureV5Optimizer.solve()` is the production entry point.
- `Instance` and `Placement` are the shared data contracts across modules.

**Chinese presenter note**

這張照流程講即可。第一步 evaluator 給 tensor，parse 成 `Instance`，裡面有 block/net/constraint lookup。接著載 checkpoint 產生 guidance。之後 candidate generation、repair、metrics、v10 proxy ranking 都使用同一份 `Instance` 和 `Placement`，所以架構比較乾淨，也方便做 no-checkpoint ablation。

---

## Slide 6 - Selectable Anchor-GNN Encoder

**One guidance contract, multiple encoders**

- MPNN: block features plus weighted block-to-block edges.
- Graph Transformer: block graph plus pin, cluster, MIB, and boundary context edges.
- Local HGT: typed block, pin, cluster, MIB, and boundary nodes with relation-specific attention.

**Shared model heads**

- `anchor`: predicted block center prior.
- `priority`: ordering signal.
- `log_aspect`: soft-block aspect-ratio prior.
- `pair_logits`: horizontal/vertical pairwise relation signal.

**Runtime adapter**

`_anchor_predictions_to_guidance()` converts the shared heads into:

```text
AnchorGuidance(rect_priors, priority, log_aspect, pairwise_axis)
```

**Chinese presenter note**

這張要強調「encoder 可換，但 downstream contract 不換」。目前 production default 是 Graph Transformer checkpoint。HGT 的定位是更忠實保留 heterogeneous floorplan graph，但它仍然只能輸出同樣的 four heads；不會直接改 decoder。這樣才可以公平比較 checkpoint 和 encoder。

---

## Slide 7 - Constraint-Aware Decoding and Repair

**Decoder**

- `relative_order.construct_relative_order_placement()` is the main constructive decoder.
- It uses anchor priors, connectivity, constraint statistics, and profile-specific ordering.
- `pairwise_axis` biases ambiguous horizontal/vertical pair relations.
- Narrow grouping pair bias adds local pressure only for ambiguous same-cluster pairs.

**Repair**

- Fixes hard geometry first: missing/overlap/area/fixed/preplaced issues.
- Repairs soft constraints: boundary snap, grouping connection, and MIB shape consistency.
- Keeps repair decisions behind hard legality and v10 proxy checks.

**Chinese presenter note**

這張可以說 decoder 是把 learned priors 轉成一個 constructive placement，不是直接相信模型座標。pairwise head 和 narrow grouping bias 只是在 ambiguous pair 上影響水平/垂直關係。repair 則先保 hard legality，再處理 boundary/grouping/MIB。這裡要避免講成「repair 任意改善 soft constraints」，因為 hard legality gate 優先。

---

## Slide 8 - V10 Ranking and Evidence-Gated Budget

**Candidate ranking**

```text
hard legality rank
  -> v10 no-runtime proxy cost
  -> soft-constraint key
  -> HPWL / bbox quality key
```

**Budget layer**

- Risk features: `score_share`, `constraint_density`, and `net_density`.
- Budget tiers: `NONE`, `LIGHT`, `MEDIUM`, `HEAVY`.
- Extra candidates, soft repair, and quality portfolio are allowed only through the budget layer.
- Conditional runtime budget stops extra work when rejected attempts or elapsed time exceed tier limits.

**Why this matters**

- Avoids soft-first choices that improve violations but damage hard feasibility or quality.
- Avoids broad portfolios or hard runtime clamps becoming defaults from single-case wins.

**Chinese presenter note**

這張是 v10 policy 的核心。先講 hard legality rank：少 block、overlap、area/fixed/preplaced violation 的 candidate 永遠優先。第二層才是 no-runtime proxy。budget layer 則決定哪些 case 值得花更多時間。它不是新 solver，而是控制 candidate/repair/portfolio 的決策面。

---

## Slide 9 - Training and Checkpoint Promotion

**Training path**

- Dataset samples and `fp_sol` are parsed into `Instance`.
- Pseudo-target logic separates clean, dirty, and repaired targets.
- Shared heads are trained with anchor, aspect, priority, and pairwise relation losses.
- Dirty samples are weighted so soft-violating golden answers do not become constraint oracles.

**Promotion path**

- Checkpoints are evaluated with full validation, not just validation loss.
- Promotion priority:
  1. `total_score_no_runtime`
  2. tail-weighted no-runtime score
  3. soft violations
  4. average runtime

**Chinese presenter note**

這張要講 training 和 production 是連接但分離的。training loss 只能說明模型學到 geometry；能不能 promotion 要看 evaluator evidence。尤其 HGT quality-first retraining 會每個 epoch 跑 evaluator，並且 evaluator-best checkpoint 只靠 full-eval evidence 選，不靠 val loss。

---

## Slide 10 - Current Evaluation Evidence

**Fresh v10 ablation surface**

| Run                                        | v10 no-runtime | v10 total | Feasible | Avg runtime | P90 runtime | Max runtime | Decision                    |
| ------------------------------------------ | -------------: | --------: | -------: | ----------: | ----------: | ----------: | --------------------------- |
| Merge-base baseline                        |         2.2184 |    2.7762 |  100/100 |       1.25s |       2.27s |       7.01s | baseline                    |
| Phase 1 v10 proxy default                  |         2.1634 |    2.7471 |  100/100 |       1.24s |       2.12s |       7.42s | keep default                |
| Phase 1 + narrow grouping pair bias        |         2.1538 |    2.6993 |  100/100 |       1.23s |       2.24s |       6.58s | promote default             |
| Phase 1 + v10 soft repair                  |         2.1446 |    2.8314 |  100/100 |       1.40s |       2.46s |       9.64s | opt-in only                 |
| Phase 1 + soft repair + conditional budget |         2.1670 |    2.7207 |  100/100 |       1.22s |       2.56s |       5.70s | runtime useful, not default |

**Evidence-backed interpretation**

- V10 proxy improves the main no-runtime metric.
- Narrow grouping pair bias is the first grouping-side change improving both primary surfaces.
- Soft repair can improve no-runtime but still carries runtime-aware risk.
- Conditional runtime budget helps runtime tail but is not the no-runtime default.

**Chinese presenter note**

這張只講有 evidence 的 claim。可以說：「我們不是看到某個 case 變好就 promote，而是看 100-case full validation。」Phase 1 v10 proxy 和 narrow grouping pair bias 是目前可以支撐 default 的；soft repair 和 conditional runtime budget 還是 ablation/opt-in，因為它們和 no-runtime/runtime surface 有 tradeoff。

---

## Slide 11 - Takeaways and Next Steps

**Takeaways**

- The solver uses learned geometry as a prior, not as a replacement for legality-aware floorplanning.
- The decoder, repair, diagnostics, and ranking layers share stable `Instance` and `Placement` contracts.
- Selectable encoders are compared through the same `AnchorGuidance` interface.
- v10 decisions are promoted only when full-validation evidence and runtime gates agree.

**Current boundary**

- Do not promote broad quality portfolios or hard runtime clamps by default.
- Do not change the decoder path for HGT without evaluator-backed evidence.
- Keep runtime optimization subordinate to no-runtime quality unless submission sanity checks fail.

**Next steps**

- Tighten soft-repair acceptance gates.
- Continue quality-first HGT retraining with evaluator-backed checkpoint promotion.
- Use validation tail diagnostics to design reusable risk triggers, not validation-ID hacks.

**Chinese presenter note**

結尾用三句話收束：第一，learned guidance + deterministic solver 是主線。第二，v10 的主指標是 full-validation no-runtime，runtime 是 gate。第三，下一步不是亂加 portfolio，而是更精準地 gating soft repair，以及讓 HGT 用 evaluator evidence 證明是否能取代 Graph Transformer default。

---

## 10-Minute Delivery Plan

|       Time | Slides | What to emphasize                                              |
| ---------: | ------ | -------------------------------------------------------------- |
|  0:00-0:35 | 1      | Project scope and main thesis.                                 |
|  0:35-1:25 | 2      | FloorSet inputs, outputs, hard/soft constraints, v10 pressure. |
|  1:25-2:10 | 3      | Learned prior plus deterministic solver principle.             |
|  2:10-3:10 | 4      | Architecture evolution and why earlier attempts changed.       |
|  3:10-4:05 | 5      | End-to-end production data flow.                               |
|  4:05-5:00 | 6      | Encoder variants and shared guidance contract.                 |
|  5:00-5:50 | 7      | Decoder and repair logic.                                      |
|  5:50-6:45 | 8      | V10 ranking and budget layer.                                  |
|  6:45-7:35 | 9      | Training and evaluator-backed checkpoint promotion.            |
|  7:35-9:05 | 10     | Evidence table and promotion decisions.                        |
| 9:05-10:00 | 11     | Takeaways, limitations, and next steps.                        |

## Short English Talk Track

Good morning. Today I will present our final project on ICCAD 2026 FloorSet Problem C, a data-driven SoC floorplanning challenge. The key idea is not to ask a neural network to directly output a final floorplan. Instead, we use the network to predict geometric priors, and we keep legality, constraint repair, and candidate ranking inside a deterministic physical-design solver.

The contest input includes block areas, net connectivity, pins, and constraints such as fixed shape, preplaced blocks, boundary requirements, grouping, and MIB constraints. The output is a rectangle for every block. Hard violations, such as overlap or wrong fixed/preplaced geometry, make a case infeasible. Soft violations affect the exponential penalty. Under the v10 scoring update, large and constraint-dense cases carry high weight, so the solver must be both legal and careful about score-risk allocation.

The architecture changed through several evidence loops. Early branches established the basic GNN-guided path, but also showed that learned geometry still needed deterministic repair. Later branches added relative-order decoding and pairwise guidance, then shifted promotion from validation loss to full-evaluator evidence. Architecture v4 separated no-runtime quality from local runtime-aware diagnostics, and Architecture v5 made the encoder selectable while preserving the downstream solver contract.

Our production path is Architecture v5. Evaluator tensors are parsed into an `Instance`, a checkpoint produces `AnchorGuidance`, candidate placements are generated by the relative-order decoder, repair passes refine the geometry, diagnostics compute placement metrics, and a v10 proxy ranker selects the best candidate. The final output is the list of `(x, y, width, height)` tuples required by the evaluator.

The neural component is intentionally modular. MPNN, Graph Transformer, and Local HGT are selectable encoders, but they all emit the same heads: anchor, priority, log-aspect, and pairwise logits. These are converted into `AnchorGuidance`, which gives the decoder rectangle priors, ordering signals, shape priors, and pairwise orientation bias. This means encoder experiments do not silently change the decoder or repair behavior.

The deterministic solver then converts guidance into a legal floorplan. The relative-order decoder uses anchor priors, connectivity, constraints, and pairwise axis signals. Repair handles overlap, boundary snapping, grouping connection, and MIB consistency. Importantly, hard legality is ranked before soft improvements, so the solver does not trade feasibility for a local soft-constraint gain.

For v10, we introduced an evidence-gated decision layer. Candidate ranking first checks hard legality, then a no-runtime proxy cost, then soft and quality keys. The budget layer uses score share, constraint density, and net density to decide whether a case deserves extra candidate or repair work. This avoids promoting broad portfolios or runtime clamps from isolated wins.

Training is also tied back to evaluator evidence. The model is trained on anchor, aspect, priority, and pairwise targets, with dirty samples weighted carefully because golden floorplans may violate soft constraints. Checkpoint promotion is not based on validation loss alone; it uses full-validation no-runtime score first, then tail score, soft violations, and runtime.

In our current v10 ablation surface, the merge-base baseline has a no-runtime score of 2.2184. The Phase 1 v10 proxy default improves it to 2.1634, and narrow grouping pair bias further improves it to 2.1538 while also improving runtime-aware total. Soft repair can reduce no-runtime score further in one setting, but it increases runtime-aware risk, so it remains opt-in.

The main takeaway is that the strongest architecture is a hybrid: learned geometric priors guide the search, deterministic physical-design logic preserves legality, and full-validation evaluator evidence decides what becomes default. The next step is to tighten soft-repair gating and continue HGT retraining under evaluator-backed checkpoint promotion.

## Backup Appendix - Prior Attempts and Why They Changed

### Appendix A - Branch / Architecture Timeline

| Branch or version                | Evidence source                                                                                                | Attempt                                                                                                           | Why it changed                                                                                                                                  |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `arch-v1`                      | git branch/log: initial `arch_new/` and checkpoint commits                                                   | Establish an initial learned floorplanning path.                                                                  | Useful starting point, but later work needed explicit solver contracts, repair, and evaluator scripts.                                          |
| `arch-v2`                      | commits adding `arch_old/`, `architecture_v2_optimizer.py`, `hetero_graph.py`, and `relative_order.py` | Add wrapper generation and more structured geometry/graph handling.                                               | The project moved toward `floorset_arch` as the stable solver package and wrapper versions as evaluator-facing labels.                        |
| `arch-v3`                      | commits adding pairwise relation head/training plumbing and GNN checkpoints                                    | Improve learned ordering and repair diagnosis.                                                                    | Checkpoint quality could not be judged by training health alone; full evaluator evidence became necessary.                                      |
| `arch-v4`                      | runtime-aware v4 design and no-runtime promotion docs                                                          | Add explicit `total_score_no_runtime`, raw runtime summaries, `.env`, and v4 wrapper.                         | Local runtime-aware score used the solver's own median runtime and could distort architecture tuning; no-runtime became the main tuning metric. |
| `v5-graph-transformer-encoder` | commits `520fb48`, `add5318`; README architecture flow                                                     | Add Graph Transformer encoder and factor-derived context edges.                                                   | Encoder changes were isolated behind the same `AnchorGuidance` contract so decoder/repair/ranking stayed comparable.                          |
| Local HGT work                   | HGT design and training commits                                                                                | Preserve typed block/pin/cluster/MIB/boundary relations with local relation-specific attention.                   | HGT remains a selectable encoder; it must beat the current checkpoint through evaluator evidence before changing production defaults.           |
| v10 branch                       | v10 proxy/runtime/grouping design and commits                                                                  | Replace soft-first acceptance with v10 proxy-first ranking, conditional runtime budget, and narrow grouping bias. | v10 evidence showed the next gains came from ranking/budget policy, not another immediate decoder or GNN rewrite.                               |

### Appendix B - Failed or Mixed-Evidence Attempts

| Attempt                       | Result                                                                                                                                                   | Decision                                                             |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Large-case candidate matrix   | With the promoted 2026-05-10 checkpoint, no-runtime total tied the default at `1.9560`; IDs 95-99 had zero no-runtime delta while runtime increased.   | Keep opt-in; do not default.                                         |
| No-Checkpoint Guidance Mode   | Feasible `100/100`, but no-runtime was far worse than configured GNN guidance (`4.6289` vs `2.0326` in the May run).                               | Use as repair diagnostic, not replacement architecture.              |
| Quality Portfolio v1          | Found ID 99 HPWL headroom, but local runtime-aware total regressed; later on Graph Transformer 0521, no-runtime regressed from `2.1194` to `2.2446`. | Keep as opt-in ablation only.                                        |
| V10 soft repair               | Improved no-runtime in one run (`2.1194 -> 2.1082`) but worsened runtime-aware total and max runtime (`6.80s -> 9.47s`).                             | Keep opt-in; pair with conditional runtime budget experiments.       |
| Hard runtime-tail clamp       | Reduced p90/max runtime but over-clamped quality; no-runtime regressed to `2.2741`, total to `3.0120`.                                               | Removed live path; use conditional runtime budget instead.           |
| Broad grouping adjacency bias | Regressed no-runtime to `2.4989` and total to `3.3102` when combined with soft repair and clamp.                                                     | Removed broad path; replaced by narrow ambiguous-pair grouping bias. |
| Narrow grouping pair bias     | Fresh v10 ablation improved no-runtime (`2.1634 -> 2.1538`) and total (`2.7471 -> 2.6993`) relative to Phase 1.                                      | Promoted default with an ablation switch.                            |

**Chinese Q&A note**

如果老師問「你們是不是一直換模型？」可以回答：不是。早期確實做過 GNN/checkpoint/encoder 嘗試，但後來的主要設計收斂成「同一個 solver contract 下比較 encoder」。v10 之後最大的改動不是換模型，而是把 candidate acceptance 從 soft-first 改成 hard-legality + no-runtime proxy-first，並用 budget layer 控制哪些 expensive path 可以跑。

## Source Anchors

- `README.md`: Architecture v5 production flow, shared output contract, active production path.
- `CONTEXT.md`: project glossary and decision relationships.
- `docs/evaluation/2026-06-05-v10-recalibration.md`: v10 score policy and ablation evidence.
- `docs/evaluation/2026-05-11-no-runtime-promotion.md`: no-runtime checkpoint promotion and large-case matrix evidence.
- `docs/evaluation/2026-05-21-no-checkpoint-guidance-baseline.md`: no-checkpoint ablation and repair headroom.
- `docs/evaluation/2026-05-21-quality-portfolio-v1.md`: quality portfolio mixed evidence.
- `docs/evaluation/2026-06-11-opt-in-flag-cleanup.md`: flag keep/remove decisions and historical cleanup.
- `docs/superpowers/specs/2026-05-11-runtime-aware-v4-design.md`: reason for v4 no-runtime/local-runtime split.
- `docs/superpowers/specs/2026-05-26-hgt-encoder-design.md`: Local HGT scope and promotion gates.
- `docs/superpowers/specs/2026-06-09-v10-budget-proxy-runtime-grouping-design.md`: v10 proxy, conditional budget, and narrow grouping rationale.
- `src/floorset_arch/optimizer.py`: `solve()`, checkpoint guidance, candidate construction, and v10 ranking.
- `src/floorset_arch/features.py`: MPNN, Graph Transformer, and HGT graph builders.
- `src/floorset_arch/nn/model.py`: selectable encoder and shared model heads.
- `src/floorset_arch/budget_layer.py`: evidence-gated budget decisions.
- `src/floorset_arch/v10_proxy.py`: hard legality and v10 proxy rank.
