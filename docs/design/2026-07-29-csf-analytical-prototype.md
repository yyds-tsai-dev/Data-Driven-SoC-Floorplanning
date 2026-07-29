# CSF analytical floorplanning prototype — design, gate protocol, and Phase-0 verdict

**Date:** 2026-07-29
**Paper:** Meng, Cheng, Chen, Xu, Chen, Zhang — *CSF: Fixed-outline Floorplanning
Based on the Conjugate Subgradient Algorithm Assisted by Q-Learning*
(arXiv:2504.03796v2), local copy `artifacts/papers/2504.03796.pdf`.
**Prototype:** `scripts/probes/csf_analytical_probe.py`
**Status:** Phase 0 (design + minimal runnable prototype) **complete**. Gate
measured. **Recommendation: do not fund the full analytical replacement.** One
narrow descendant bet survives (§7).

---

## 1. What the paper actually proposes

CSF is a two-stage FA-AOM (analytic-model floorplanner) whose distinguishing
choice is that it does **not** smooth the objective. Both stages are solved by a
conjugate subgradient algorithm (CSA) whose scalar step-size factor `c` is
scheduled by tabular Q-learning (CSAQ).

### 1.1 Model (paper §3.1)

Decision variables are module **centers** `x = (x_1..x_n)`, `y = (y_1..y_n)`;
widths `ŵ_i` and heights `ĥ_i` are inputs (the benchmark modules are hard).

    min W(x,y)   s.t.  D(x,y) = 0,  B(x,y) = 0                          (Eq. 1)

- **Wirelength** (Eq. 2) — per-net half-perimeter over module centers:
  `W = Σ_{e∈E} (max_{v∈e} x − min_{v∈e} x + max_{v∈e} y − min_{v∈e} y)`.
- **Overlap** (Eq. 3–5) — sum of pairwise intersection areas:
  `D = Σ_{i,j} O_ij(x)·O_ij(y)`, with the axis overlap length

      O_ij(x) = min(ŵ_i, ŵ_j)                     if |Δx| ≤ |ŵ_i−ŵ_j|/2
              = (ŵ_i − 2|Δx| + ŵ_j)/2             if |ŵ_i−ŵ_j|/2 < |Δx| ≤ (ŵ_i+ŵ_j)/2
              = 0                                  otherwise

  which is exactly `clip((ŵ_i+ŵ_j)/2 − |Δx|, 0, min(ŵ_i,ŵ_j))` — the form the
  prototype uses, since it vectorises to one dense `n×n` expression.
- **Boundary violation** (Eq. 7) — `B = Σ_i (b_{1,i}(x)+b_{2,i}(x)+b_{1,i}(y)+b_{2,i}(y))`
  with `b_1(x)=max(0, ŵ_i/2 − x_i)`, `b_2(x)=max(0, ŵ_i/2 + x_i − W*)`, and the
  fixed outline sized by

      W* = sqrt((1+γ)·A/R),   H* = sqrt((1+γ)·A·R)                       (Eq. 6)

  where `A = Σ areas`, `R` the width–height ratio, `γ` the whitespace ratio.
- **Global stage** minimises `f_g = α·W + λ·D + μ·B` (Eq. 8); **legalization**
  minimises `f_l = λ·D + μ·B` (Eq. 9).

### 1.2 CSA (paper Algorithm 1)

    g_0 ∈ ∂f(u_0),  d_0 = 0
    for k = 1..k_max:
        g_k ∈ ∂f(u_{k-1})
        η_k = g_kᵀ(g_k − g_{k−1}) / ‖g_{k−1}‖²        (Polak–Ribière)
        d_k = −g_k + η_k d_{k−1}
        â_k = c / ‖d_k‖₂                              (scale-free step)
        u_k = u_{k−1} + â_k d_k

The step length is **not** a line search: it is `c` divided by the direction
norm, so `c` is literally "how far a block moves per iteration". That is why the
paper spends its ML budget on scheduling `c`.

### 1.3 Q-learning over `c` (paper §3.3, Alg. 3)

A `p×m` Q-table (paper uses `p = m = 5`); state = individual in the population,
action = one candidate value of `c`. Action sampling is `P_i^j = Q(s_i,a_j)/Σ_l
Q(s_i,a_l)` (Eq. 12); reward `R = (f_pre − f_post)/100` (Eq. 13); update
`Q^k = (1−α_0)Q^{k−1} + α_0(R + γ_0 Q_max)` (Eq. 10) with `α_0=0.4, γ_0=0.8,
m_0=0.6`. `c` is refreshed every `k_t` generations.

### 1.4 Outer loops

- `GFloorplan` (Alg. 4): run CSAQ, then `λ ← 1.3·λ`, repeat until total overlap
  `< A/v`.
- `CSF` (Alg. 2): LHS-sample a population of `p` initial layouts inside the
  outline; global stage → legalization; if illegal, randomly rotate modules and
  retry, up to `t_max = 10`.
- Legalization: either `ILA-CG` (Alg. 5–6, constraint-graph critical-relationship
  surgery to shrink the layout into the outline) or `LA-CSAQ` (Alg. 7, CSA
  directly on `f_l`, then a CG pass to compact). Published result: LA-CSAQ wins
  on wirelength, runtime and success rate (Table 6).

### 1.5 Published hyper-parameters (Table 3, GSRC)

| stage | optimizer | k_t | k_max | α | λ | μ | c_0 | v |
|---|---|---|---|---|---|---|---|---|
| global | CSAQ | 40 | 200 | 1 | 20 | 100 | 25 | 100 |
| legalize | CSAQ | 100 | 5000 | – | 1 | 10 | 10 | – |

Action sets for `c`: global `[8,12,15,20,25]`, legalization `[0.1,0.8,5,10,20]`.
Runtime reference: n300 (300 hard modules) ≈ 12.4 s total in C++ on a laptop.

---

## 2. What transfers, what does not, and what is missing

### 2.1 Simplifications this problem grants us

| Paper | FloorSet ICCAD-2026 | Consequence |
|---|---|---|
| Per-net max/min HPWL (nonsmooth, needs net enumeration) | `b2b` is a list of **weighted pairwise edges**; `p2b` anchors a block to a **fixed pin coordinate** (`iccad2026_evaluate.py:153-194`) | `W` is **convex piecewise-linear** in the centers. Subgradient is exactly `w_ij·sign(Δ)`. Strictly easier than the paper. |
| I/O pad assignment | Pins are given at fixed coordinates | **Not needed.** Drop entirely. |
| Hard modules only (benchmark restriction) | Soft blocks: any `(w,h)` with `\|w·h − a\|/a ≤ 0.01` | Big *extra* degree of freedom the paper never uses — and, per §5, the one that actually decides the score. |
| Module rotation as the retry move (Alg. 2 line 9) | Subsumed by free aspect choice | Replace rotation with aspect resampling. |

### 2.2 The fixed outline must be *synthesized*

FloorSet has **no given outline**. `Area_bbox_gap` is measured against the
*golden* layout's bbox (`iccad2026_evaluate.py:197-207, 370-373`). So `W*,H*`
become design parameters. Measured calibration over the 20 tail cases
(n = 101..120):

- golden whitespace `1 − ΣA/bbox` = **2.8 – 4.6 %** (mean 3.4 %) — the golden
  layouts are rectilinear polygons packed essentially solid;
- golden outline aspect `W/H` = **0.49 – 0.80** (mean ≈ 0.60), *not* square;
- the **pin bounding box** is a tight, reliable predictor of that aspect
  (e.g. case 99: pin box 147×313 → R 0.47, golden 131×265 → R 0.494).

Hence the prototype sets `R = pin_H/pin_W` and `γ ≈ 0.095`, which is the
whitespace the current column backbone actually achieves (its `area_gap ≈ 0.06`
over a golden that already has 3.4 % → ≈ 9.5 % whitespace). **Note that the
paper's benchmarks run at γ = 15 %; this problem is a materially tighter regime
than anything CSF was validated on.**

### 2.3 Constraints the paper does not model at all

FloorSet's soft-violation term is `exp(2·V_rel)` with
`V_rel = (V_boundary + V_grouping + V_mib)/N_soft`
(`iccad2026_evaluate.py:498-549`). None of the three appears in CSF:

- **boundary** — block must *touch* a specified edge/corner of the final bbox
  (bitmask 1=L,2=R,4=T,8=B). Tail cases carry **26–37 boundary blocks each**
  (~30 % of all blocks).
- **grouping** — same-cluster blocks must form **one edge-connected component**
  (shapely `unary_union`); 3–4 groups per tail case, 14–31 slack units.
- **MIB** — all members of a MIB group must share an identical `(w,h)` rounded
  to 4 dp.

`N_soft` on the tail is 43–67. Planned handling in the prototype:
- **MIB: free by construction** — give every MIB member the same square shape
  when their target areas agree (they do), which also keeps exact area.
- **boundary / grouping: continuous surrogates only** (`--w-boundary`,
  `--w-group`): an L1 pull of the constrained edge toward the outline edge, and
  an L1 attraction between same-cluster pairs. These are *proximity* proxies;
  the constraints are *abutment/touching* predicates that only a discrete
  legalizer can actually satisfy. This is the acknowledged weak point and §5
  shows it is fatal.

---

## 3. Prototype architecture (`scripts/probes/csf_analytical_probe.py`)

Pure CPU, NumPy only, no torch autograd (the subgradient is analytic).

```
build_analytic_instance()   Instance -> dense arrays; frozen shapes with w*h == a
                            exactly; outline (W*,H*) from (gamma, pin-box R);
                            preplaced blocks marked `frozen` (centers pinned)
objective()                 f, dF/dcx, dF/dcy, total_overlap
                            W  : evaluator-faithful weighted-L1 over edges+pins
                            D  : dense n x n clip((w_i+w_j)/2-|dx|, 0, min(w_i,w_j))
                            B  : paper Eq. 7 against (W*,H*)
                            +  : optional boundary-pull / cluster-attraction
csa()                       paper Alg. 1 verbatim (Polak-Ribiere, c/||d|| step)
global_floorplan()          paper Alg. 3 + 4: population of p, Q-table over c,
                            lambda <- 1.3*lambda until overlap < A/v
legalize_cg()               HCG/VCG per paper Sec. 3.4 rules 1-3, then
                            longest-path packing; preplaced blocks are graph
                            SOURCES so nothing can push them off spec
solve_case()                warm start -> global -> legalize -> CG re-pack
                            (paper Fig. 1 compression), bbox-monotone
```

**Hard-legality story.** Overlap-free and exact-area are **by construction**:
shapes never change after `build_analytic_instance`, and the longest-path pack
over a complete CG pair separates every block pair on at least one axis.
Preplaced position is preserved by making preplaced blocks sources in both
graphs. Fixed-shape blocks keep spec dimensions. This is why the smoke runs
report `overlap_violations = 0`, `area_violations = 0`,
`dimension_violations = 0` — the prototype is hard-legal; it just scores badly.

**Interfaces.** The probe deliberately does **not** touch
`src/floorset_arch/`. It reads the validation set through
`litetestLoader.FloorplanDatasetLiteTest` and scores through the evaluator's own
`evaluate_solution(..., runtime=1.0, median_runtime=1.0)`, so the reported
number is the **No-Runtime Quality Score** contribution, evaluator-faithful.
Column-backbone reference costs are read from
`artifacts/partner_eval/budget35_dm2_df650k_s10.json` (the 3.5 s-budget
champion, `total_score_no_runtime` 1.1401), so the head-to-head is at matched
per-case wall clock without re-running the backbone.

Had the gate passed, the production interface would have been a **portfolio
sibling**, not a seed: `optimizer.solve()` would run the analytical pipeline and
the column backbone under a split budget and select with the V10 proxy. It would
**not** have been wired as a seed into `legalize_rectangles`, because the seed
channel is already measured dead — GT-coordinate seeds move the total by
< 0.002 (`scripts/probes/gt_seed_optimizer.py`).

---

## 4. Gate protocol

- **Subset:** all validation cases with `n > 100` (20 cases, ids 80–99). They
  carry **81.1 %** of the total weighted score (`λ_i ∝ e^{n_i/12}`).
- **Budget:** 3.5 s per case, matching the current champion configuration.
- **Metric:** weighted no-runtime cost over the subset,
  `Σ λ_i·cost_i / Σ λ_i`, computed by the official `evaluate_solution` with
  `runtime_factor = 1`.
- **Reference:** same 20 cases from `budget35_dm2_df650k_s10.json`
  (subset weighted no-runtime **0.9124** as a share of the full total; **1.1246**
  as a subset-normalised weighted mean).
- **Pass:** `delta ≤ −0.0100`. Anything else → record the verdict and stop.
- **Command:**
  `uv run scripts/probes/csf_analytical_probe.py --tail --budget 3.5 --output <json>`

### 4.1 Headroom arithmetic (computed before running the prototype)

Oracle deltas on the **full 100-case** no-runtime total (baseline 1.1401),
applying the oracle only to the `n>100` subset:

| oracle applied to the tail | total | delta |
|---|---|---|
| `hpwl_gap → 0` | 1.1182 | **−0.0219** |
| `area_gap → 0` | 1.1145 | **−0.0256** |
| `V_rel → 0` | 1.0837 | **−0.0565** |
| `hpwl_gap` halved | 1.1292 | −0.0110 |
| `hpwl→0` **and** `area→0`, but `V_rel = 0.10` | 1.2187 | **+0.0785** |
| `hpwl→0` **and** `area→0`, but `V_rel = 0.15` | 1.3229 | **+0.1827** |

Current tail means: `hpwl_gap` 0.063, `area_gap` 0.060, `V_rel` 0.033.

**This table is the whole story.** The soft-violation channel is worth more than
HPWL and area *combined*, and it is exponential: a perfect-wirelength,
perfect-area analytical solver that lets `V_rel` drift to only 0.10 **loses by
+0.0785**. Any analytical replacement must therefore reproduce the backbone's
`V_rel ≈ 0.033` before its wirelength advantage counts for anything.

---

## 5. Phase-0 results — the gate fails, and the failure is structural

### 5.1 Prototype vs column backbone — full 20-case gate run (3.5 s/case)

`uv run scripts/probes/csf_analytical_probe.py --tail --budget 3.5`

    id    n      CSF      COL    delta    hpwlg    areag    Vrel  feas ovl dim
    80  101  10.0000   1.1606  +8.8394   1.2009   2.5372  0.8889  True   0   0
    81  102  10.0000   1.1018  +8.8982   0.9950   0.7443  0.9310  True   0   0
    82  103  10.0000   1.1188  +8.8812   1.6659   2.1845  0.7458  True   0   0
    83  104  10.0000   1.2092  +8.7908   1.4130   2.2168  0.8364  True   0   0
    84  105  10.0000   1.1498  +8.8502   0.9921   0.7467  0.8548  True   0   0
    85  106  10.0000   1.0817  +8.9183   1.2174   2.2133  0.7812  True   0   0
    86  107  10.0000   1.1130  +8.8870   1.7021   1.9070  0.8364  True   0   0
    87  108   9.9629   1.0423  +8.9206   0.9069   1.0053  0.8140  True   0   0
    88  109  10.0000   1.2072  +8.7928   1.9819   3.7411  0.7937  True   0   0
    89  110  10.0000   1.3417  +8.6583   2.5276   3.6219  0.8491  True   0   0
    90  111  10.0000   1.2959  +8.7041   1.4950   3.5700  0.7879  True   0   0
    91  112   8.5914   1.1273  +7.4640   0.7421   0.6560  0.8103  True   0   0
    92  113  10.0000   1.1136  +8.8864   2.3907   1.3460  0.8571  True   0   0
    93  114  10.0000   1.1197  +8.8803   1.2763   2.4549  0.9032  True   0   0
    94  115  10.0000   1.0708  +8.9292   1.9579   1.8531  0.8154  True   0   0
    95  116  10.0000   1.0813  +8.9187   2.9978   3.0760  0.8182  True   0   0
    96  117  10.0000   1.0566  +8.9434   0.8380   0.9804  0.8615  True   0   0
    97  118  10.0000   1.0566  +8.9434   1.4892   2.3903  0.8000  True   0   0
    98  119  10.0000   1.0733  +8.9267   1.7795   3.4186  0.8269  True   0   0
    99  120  10.0000   1.1656  +8.8344   1.2214   3.7351  0.7164  True   0   0

    subset weighted no-runtime:  CSF 9.9274   COLUMN 1.1246   delta +8.8027

Hard legality holds on **20/20 cases** (0 overlaps / 0 area / 0 dimension
violations) — the by-construction argument in §3 checks out. But `V_rel` lands
at **0.72–0.93** (`exp(2·0.86) ≈ 5.6×`) and `area_gap` at **0.66–3.74**, so 19 of
20 cases saturate the feasible cost cap (9.999999).

### 5.2 Where the loss lives — golden-coordinate ablation

Feeding the **golden** block centers straight into the prototype's legalizer
removes the analytical stage from the equation entirely:

| case | shapes | `hpwl_gap` | `area_gap` | `V_rel` | cost |
|---|---|---|---|---|---|
| 81 (n=102) | square, exact-area | 0.2513 | 0.5675 | 0.6379 | 5.05 |
| 81 | **golden bbox shapes** | 0.0881 | 0.1507 | 0.4483 | 2.74 |
| 96 (n=117) | square, exact-area | 0.1936 | 0.5558 | 0.7538 | 6.21 |
| 96 | **golden bbox shapes** | 0.0433 | 0.1900 | 0.3692 | 2.34 |
| 99 (n=120) | square, exact-area | 1.0232 | 3.2968 | 0.6866 | 10.0 |
| 99 | **golden bbox shapes** | 0.8829 | 2.8197 | 0.3881 | 6.20 |

Compare the column backbone on the same cases: `area_gap` 0.0023 / 0.0341 /
0.0534 and `V_rel` 0.0000 / 0.0154 / 0.0448.

Three conclusions, in order of importance:

1. **The CG-pair + longest-path legalizer is the dominant loss, not the
   analytical stage.** With *perfect* centers *and* golden shapes it still
   inflates `area_gap` from 0.002 → 0.151 (case 81) and 0.034 → 0.190
   (case 96). CSF's legalization leg is simply a weaker legalizer for this
   objective than the column-slicing parallel-restart SA already in production.
2. **Freezing shapes costs ~0.37–0.40 of `area_gap` on its own** (square vs
   golden-bbox rows). The paper's model has no shape variables because its
   benchmarks are hard-module; FloorSet's decisive lever is exact-area **aspect
   flexibility**, which is exactly what the column backbone exploits per column.
   An analytical formulation would need `(w,h)` as optimization variables under
   the hyperbolic constraint `w·h = a` — a substantial extension the paper does
   not provide.
3. **Soft constraints stay catastrophic even with golden coordinates**
   (`V_rel` 0.37–0.45), because boundary/grouping are abutment predicates that
   survive only if the *legalizer* is built to preserve them. Continuous pull
   terms cannot deliver them.

Case 99 (n=120, `b2b` is a near-complete graph with 7056 edges) degenerates
completely — rule-3 tie-breaking produces a pathological chain. That is a fixable
prototype bug, but fixing it changes nothing about the verdict.

### 5.3 Verdict

Gate threshold `−0.0100`; measured **`+8.8027`** for the prototype (20/20 cases) and, as an
*upper bound* on any amount of analytical-stage tuning, `+1.3 … +5.1` per case
from the golden-coordinate oracle. The gap is not a tuning gap. **Close the
"analytical replacement for the column backbone" line.**

---

## 6. Honest scope statement

What was **not** verified:

- I did not implement the paper's `ILA-CG` critical-relationship surgery
  (Alg. 5–6) or `LA-CSAQ` (Alg. 7). My legalizer is the simpler longest-path
  pack. The paper reports LA-CSAQ beating plain CG legalization by ~1–36 % on
  wirelength — that is a *wirelength* margin, roughly one order of magnitude
  smaller than the `area_gap` and `V_rel` gaps measured in §5.2, so it does not
  change the verdict, but the "+1.3…+5.1 oracle" number would improve somewhat
  under a full ILA-CG/LA-CSAQ implementation.
- The Q-learning scheduler is implemented but its benefit was never isolated;
  the paper itself reports it as significant on only 1 of 5 circuits for the
  global stage (Table 4, `p`-values).
- I did not attempt an aspect-variable analytical formulation (`w·h = a`
  eliminated as `w = sqrt(a·e^r)`, `h = sqrt(a/e^r)`); §5.2 says this is where a
  serious analytical attempt would have to start.
- The reference costs come from a stored evaluation JSON, not a re-run of the
  backbone in the same process. The two smoke cases' reference numbers were
  spot-consistent with the stored table.

---

## 7. The one surviving descendant bet

The measurement that kills the replacement also points at a much cheaper
descendant: **CSA as the coordinate solver inside the existing refiner.**

Rationale. At a *fixed topology* — the constraint DAG the column backbone
already produces — the FloorSet HPWL is convex piecewise-linear in the
coordinates, so a conjugate-subgradient sweep is a principled solver for exactly
the problem `src/floorset_arch/refine/slack_solve.py` currently attacks with
sequential weighted-median projected sweeps. That path inherits the backbone's
`V_rel` and `area_gap` untouched (the refiner is guard-gated and
failure-contained by `refine/api.py`), so it is not exposed to the exponential
violation term at all.

- **Ceiling:** `hpwl_gap → 0` on the tail = **−0.0219** total. Clearing the
  −0.01 gate requires capturing ≈ 46 % of the residual HPWL gap.
- **Cost:** low — the objective, the analytic subgradient and the CSA loop are
  already written and tested in `csf_analytical_probe.py:objective/csa`; only
  the DAG-projection step is new.
- **Risk:** the refiner is already an L1-optimal-per-sweep method, so the
  realistic capture may be well under 46 %. Measure on the same 20-case tail
  with the same −0.01 gate before funding anything further.

Everything else from this paper — the fixed-outline model, the CG legalization,
the population/Q-learning layer — should be considered closed for this contest.

---

## 8. Engineering-effort estimate (recorded for completeness; Phase 1+ not funded)

| phase | scope | est. |
|---|---|---|
| 0 (done) | model transcription, prototype, gate measurement | 1 day |
| 1 | aspect variables (`w·h = a` eliminated), ILA-CG + LA-CSAQ legalization | 4–6 days |
| 2 | boundary/grouping-aware legalization to reach `V_rel ≤ 0.05` | 5–8 days, high risk |
| 3 | portfolio integration + budget-layer gating | 2 days |

Phases 1–2 would have to deliver *simultaneously* to reach parity, and parity is
not the bar — the bar is −0.01. Not fundable against the remaining contest
calendar.
