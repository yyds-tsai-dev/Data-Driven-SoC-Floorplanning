# Backend External Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a source-grounded Chinese backend-method report at `docs/research/後端報告.md` that matches the structure, tone, and evidence discipline of `docs/research/前端報告.md` for an external technical-exchange audience.

**Architecture:** Treat the report as one independently reviewable documentation deliverable. Establish the production call chain and a claim-evidence ledger first, draft the twelve approved sections from mechanism and evaluator evidence, then verify terminology, defaults, metrics, Markdown integrity, and Graphify synchronization.

**Tech Stack:** Markdown, Python 3.12 source inspection, `pytest`, FloorSet evaluator records, Graphify, Git.

## Global Constraints

- Do not modify solver, tests, checkpoints, evaluator behavior, or `docs/research/前端報告.md`.
- Describe current production code before tests, sealed experiment records, README prose, or historical documents when sources disagree.
- Do not disclose checkpoint files, internal tuning scripts, unsealed strategies, or scratchpad-only numbers.
- Distinguish production defaults from opt-in, legacy, and rejected paths.
- Every number and strong conclusion must identify its evaluation set, metric, comparator, and runtime condition where applicable.
- Preserve unrelated user changes in the shared worktree.

---

### Task 1: Produce and verify the backend external-exchange report

**Files:**
- Create: `docs/research/後端報告.md`
- Read: `docs/research/前端報告.md`
- Read: `src/architecture_v5_optimizer.py`
- Read: `src/floorset_arch/optimizer.py`
- Read: `src/floorset_arch/parser.py`
- Read: `src/floorset_arch/models.py`
- Read: `src/floorset_arch/relative_order.py`
- Read: `src/floorset_arch/repair.py`
- Read: `src/floorset_arch/budget_layer.py`
- Read: `src/floorset_arch/risk_budget.py`
- Read: `src/floorset_arch/quality_portfolio.py`
- Read: `src/floorset_arch/diagnostics.py`
- Read: `src/floorset_arch/v10_proxy.py`
- Read: `README.md`
- Read: sealed evidence under `docs/experiments/`, `docs/design/`, and tracked evaluator artifacts selected by the searches below
- Update: `graphify-out/` through the repository-required incremental Graphify command

**Interfaces:**
- Consumes: evaluator tensors, current `ArchitectureV5Optimizer.solve()` behavior, constraint and ranking contracts, official100 evaluator evidence, and the approved report design in `docs/superpowers/specs/2026-08-20-backend-external-report-design.md`.
- Produces: one Chinese Markdown report whose public contract is the twelve-section structure and whose claims trace to repository evidence.

- [ ] **Step 1: Re-establish the scoped architecture context**

Run:

```bash
graphify query "後端求解器從 evaluator input 經 Instance、anchor guidance、candidate portfolio、relative-order placement、repair、quality refinement、V10 proxy 到輸出的 production data flow" --budget 7000
```

Expected: output includes `ArchitectureV5Optimizer.solve()`, `parse_instance()`, `CandidateSpec`, `construct_relative_order_placement()`, `repair_placement()`, `refine_quality_candidate()`, `placement_metrics()`, and `v10_proxy_rank()` or their current equivalents. If a named node is absent, inspect the current source rather than inferring the missing link.

- [ ] **Step 2: Locate current-code and sealed-evidence sources**

Run semantic searches first:

```bash
semble search "ArchitectureV5Optimizer candidate selection repair quality portfolio V10 proxy" . --content all --top-k 20
semble search "official100 feasible 100 area gap hpwl gap utilization runtime backend" . --content all --top-k 30
semble search "backend experiments rejected disproven no gain paired evaluation" . --content docs --top-k 30
```

Then confirm exact symbols and metrics with exhaustive literal searches:

```bash
rg -n "def (solve|_candidate_specs|_build_candidate|_repair_candidate|_candidate_rank|_try_anchor_guidance)|construct_relative_order_placement|repair_placement|refine_quality_candidate|v10_proxy_rank" src/floorset_arch src/architecture_v5_optimizer.py
rg -n "official100|100/100|area gap|hpwl gap|utilization|利用率|RuntimeFactor|total_score_no_runtime|disproven|rejected|無效|判死" README.md docs/experiments docs/design artifacts docs/research/前端報告.md
```

Expected: a bounded set of current source definitions and tracked evidence documents. Ignore matches from `scratchpad/`, generated submission copies, and untracked files.

- [ ] **Step 3: Verify behavior contracts before writing claims**

Read the exact source spans returned by Step 2, then run:

```bash
uv run pytest tests/test_optimizer.py tests/test_budget_layer.py tests/test_quality_portfolio.py tests/test_v10_proxy.py -q
```

Expected: all selected tests pass. If a named test file does not exist, use `rg --files tests | sort` to identify the current equivalent and record the substitution in the final handoff. A failing test blocks any claim that depends on that behavior but does not authorize a solver change.

- [ ] **Step 4: Build the internal claim-evidence ledger**

Before drafting prose, record each major claim in working notes using this exact schema:

```text
Claim | Source path and line/symbol | Evaluation set | Metric/condition | Status
production call chain | src/floorset_arch/optimizer.py::ArchitectureV5Optimizer.solve | n/a | current code | supported
legality-first ranking | src/floorset_arch/optimizer.py::_candidate_rank + src/floorset_arch/v10_proxy.py::v10_proxy_rank | n/a | current code/tests | supported
100/100 feasibility | tracked official100 evaluator record | official100 | feasible cases, runtime condition stated | supported or omit
area is the residual bottleneck | tracked official100 evaluator record + 前端報告.md section 11 | official100 | area gap contribution, runtime condition stated | supported or soften
negative-result claim | sealed experiment document | named case set | paired delta/statistical condition | supported or omit
```

Expected: every quantitative or causal statement planned for the report is `supported`; any row marked `inferred`, `conflicted`, or `missing` must be softened to a bounded qualitative statement or removed.

- [ ] **Step 5: Draft the report with the approved section contract**

Create `docs/research/後端報告.md` with this exact top-level structure:

```markdown
# ICCAD 2026 Problem C — 後端方法報告

一份給外部交換用的說明，涵蓋我們的後端如何把候選版圖轉成合法解、如何在候選間分配計算與選解、實測到什麼，以及哪些方向已被我們自己的量測否定。

---

## 1. 一句話總結
## 2. 後端的任務邊界
## 3. 統一問題表示：`Instance` 與 `Placement`
## 4. Guidance 是偏置，不是答案
## 5. Candidate portfolio：不押單一路徑
## 6. Relative-order 建構式放置
## 7. Repair 與 quality refinement
## 8. Hard-legality gate 與 V10 proxy
## 9. 後端的實測品質
## 10. 我們試過並被自己的量測否定的方向
## 11. 給交換對象的三個要點
## 12. 我們目前的位置
```

Within sections 3–8, use the module triad `motivation -> mechanism -> evidence/role`. Include one terminal-readable pipeline block from evaluator tensors to returned `(x, y, w, h)` rows. In sections 9–10, attach the case set and runtime condition to every result. In section 12, state the evidence boundary and avoid new results.

- [ ] **Step 6: Enforce terminology and disclosure boundaries**

Run:

```bash
rg -n "ensemble|V10 total score|等同官方|一定|永遠|首次|最佳|革命|checkpoint\.pt|scratchpad|FLOORSET_" docs/research/後端報告.md
rg -n "production|default|opt-in|legacy|歷史|可選|預設" docs/research/後端報告.md
```

Expected: no terminology collision (`candidate portfolio` must not become `ensemble`), no claim that V10 no-runtime proxy equals the official total score, no unsupported universal or novelty language, no checkpoint path or scratchpad disclosure, and every optional or historical path is explicitly labelled. Any `FLOORSET_` occurrence must be necessary for public mechanism comprehension or removed.

- [ ] **Step 7: Verify metrics, structure, and Markdown integrity**

Run:

```bash
rg -n "[0-9]+([.][0-9]+)?%?|official100|100/100|RuntimeFactor|runtime|area|HPWL" docs/research/後端報告.md
rg -n '^## [0-9]+[.] ' docs/research/後端報告.md
rg -n 'T(BD)|T(ODO)|FIX(ME)|待(定|補)|\[Evidence (needed|missing)' docs/research/後端報告.md
git diff --check -- docs/research/後端報告.md
```

Expected: each numeric match maps to a supported ledger row; exactly twelve numbered top-level sections appear; the placeholder scan returns no matches; `git diff --check` returns no output.

- [ ] **Step 8: Synchronize Graphify and inspect the final diff**

Run:

```bash
graphify update .
git status --short
git diff -- docs/research/後端報告.md
git diff --check
```

Expected: Graphify completes without an extraction error; the report diff contains only the approved external document; unrelated pre-existing worktree changes remain untouched. Graphify artifacts may update as expected and must not be used as evidence for claims that were not already supported by source files.

- [ ] **Step 9: Commit the report without unrelated files**

Run:

```bash
git add -- docs/research/後端報告.md
git commit -m "docs: add backend exchange report"
```

Expected: one focused commit containing only `docs/research/後端報告.md`. Leave unrelated user files and Graphify-generated changes unstaged unless the user explicitly asks to commit them.
