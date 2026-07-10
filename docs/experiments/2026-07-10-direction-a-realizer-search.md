# 2026-07-10 — Direction A: Realizer-Faithful Decoder + (order,shape) Search — CLOSED

**Direction A is closed (disproved, not merely "failed to find").** The goal was
to open a search / analytic-hint route that *beats* the production column-backbone
on the weighted tail (n≥100 = 77% of the prize pool) by feeding good geometry
through an order-faithful exact decoder. Across seven gates it does not: the tail
has **no headroom** these realizers can capture. Production is already at the
quality ceiling reachable by preserve-or-compact realization; the apparent tail
"gaps" were realizer self-damage, not distance to a better layout.

Two artifacts are **keepers** (portfolio-safe, no promotion claim): the *faithful*
realizer (identity on a good hint) and the *anchored* realizer (non-destructive on
legal hints, compaction-capable on illegal ones). Neither is a search engine.

All numbers are offline `score_case` cost / `compute_total_score` no-runtime total
(lower is better), paired against the gate0 production-layout cache
(`prod_layouts_gate0.json`, baseline total **1.2724**; **old-config cache** — any
deploy claim needs a fresh production rerun). Everything lives in
`scripts/probes/`; `src/` and `FloorSet/` were never edited.

## Instruments

- `scripts/probes/gen_decoder_probe.py` — the exact decoder. Three realize modes,
  all opt-in; `compact` (legacy) and `faithful` are frozen byte-identical
  (regression: compact **1.6535**, faithful **1.2724**, wins/losses unchanged).
  - `compact`: ASAP longest-path recompaction (+ repair/refine). Legacy.
  - `faithful`: keep-valid-hint-shapes + geo-faithful axis extraction + longest
    path **floored at the hint coords** ⇒ a legal exact-area hint round-trips to
    itself (tax +0.000). A fidelity realizer; a **ratchet** (monotone-up, cannot
    compact).
  - `anchored`: keep-shapes + geo-axis, but two-sided — a skeleton (ASAP for an
    illegal hint, or the **legal hint itself** when legal) + `refine.slack_solve.
    project_axis` toward net + rescaled-hint + wall-target anchors, inside
    `build_axis_dags`'s constraint graph. Non-destructive on legal hints,
    compaction-capable on illegal ones. Repair/evict floor enabled implicitly.
- `scripts/probes/analytic_hints.py` — QP global placement (b2b clique + p2b pin
  pulls + wall anchors + grid reg), light de-overlap, exact-area square/MIB
  shapes; realized via `--realize {faithful,anchored}`.
- `scripts/probes/perturb_probe.py` — perturbation-smoothness probe. Move classes
  order-swap / centroid-swap / shape-perturb (exact area). `--realize` selects the
  realizer under test. Δcost vs the realizer's own seed (`reflow(base)`).
- `scripts/probes/anchored_sa_pilot.py` — greedy+low-T SA over anchored; genotype
  = realized layout, objective = exact `score_case`; seed protection; centroid
  dropped. `--case-ids`/`--n-seeds` for the confirmation run.

Cost model (verified exact): `cost = (1 + 0.5·(hpwl_gap + area_gap)) · exp(2·v_rel)`
— quality and violations are **multiplicative** co-factors.

## Gate-by-gate

| Gate | What | Result | Verdict |
|---|---|---|---|
| **1 fidelity** | `faithful` on production hints | 1.6535 → **1.2724** (tax **+0.000**, 100/100, ties 100/100); golden → **1.1079** (identity) | pass (fidelity) |
| **P1 search** | perturb `faithful` | tail improving **1.4%**, best **−0.0002**, tail median\|Δ\| **0.36**, **0/185** bbox shrink | **NO-GO** (ratchet) |
| **P2 analytic** | `analytic_hints` → `faithful` | **9.95**, 0/100 wins, 58/100 shelf; placer HPWL only +21–34% but realization inflates tail HPWL 1.21→1.92, area 1.56→2.10 | **NO-GO** |
| **A1 fidelity** | `anchored` on production | wall-anchored 1.606 (4 wins) → **legal-hint skeleton 1.2724** (96 ties, 1 win) | non-destructive |
| **A2 analytic** | `analytic_hints` → `anchored` | **9.95 → 6.04** (→5.78 tuned), shelf **58 → 0**, tail HPWL 1.92→1.44, v_rel 0.86→0.57, 1 win | placer-bound |
| **A3 search** | perturb `anchored` | tail improving **17%**, best **−1.0**, shape moves smooth (med\|Δ\| **0.067**); pooled tail median\|Δ\| 0.21; **0/315** bbox shrink | qualified GO |
| **Pilot v1** | SA×10 (200 mv) | 4 small wins (pre-existing A1 edges); **tail 0 wins**, plateau **+0.035…+0.144**; idx95 seed 2.23→best 1.29 (+0.94 gain, still +0.11 > prod) | tail stall |
| **Pilot v2** | SA×10, legal-hint skel + seed-prot + no centroid | **tail 0 wins**, all vs_base **0.000**, gain **+0.0000** everywhere (86–99 accepts) | stall (hollow plateau) |
| **Confirmation** | test99/95/97/94 × 3 seeds, ≤1200 mv/150 s | **11/12 ties, 1 float-noise "win"** (idx95 s2 −0.0001); best vs_base across all = **−0.0001** | DISPROVED |

## Confirmation run (evidence fixation: "failed to find" → "disproved")

test99, test95 (required) + test97, test94, **3 RNG seeds each, up to 1200 moves
or 150 s/seed**, shape+order moves, seed protection on. Per (case, seed):

| case | n | seeds: best − base | moves (mv) / accepted (acc) | wins |
|---|---|---|---|---|
| 99 | 120 | 0.0000 / 0.0000 / 0.0000 | 172·0, 64·1, 40·0 | 0 |
| 95 | 116 | 0.0000 / 0.0000 / **−0.0001** | 58·3, 74·4, 53·4 | 1 (noise) |
| 97 | 118 | 0.0000 / 0.0000 / 0.0000 | 65·2, 59·1, 212·2 | 0 |
| 94 | 115 | 0.0000 / 0.0000 / 0.0000 | 726·154, 353·43, 742·**275** | 0 |

**11/12 ties; the single sub-base result is −0.0001** (idx95 seed2, one accepted
move — HPWL micro-jitter at the float-noise floor). idx94 accepted **154 / 43 /
275** moves across seeds with **zero** improvement — extensive exploration, no
downhill. 🔴 The −0.0001 (and a −0.0003 seen in an earlier uncapped run) are
FLAGGED as sub-base results per protocol, but they are ≥2 orders of magnitude
below the GO bar (≥2/6 *meaningful* tail wins) and require many moves to stumble
onto; they **confirm** rather than refute the close. Verdict: **no meaningful
tail headroom exists** — direction A is disproved, not merely unsearched.

## Death mechanism

1. **The tail gap was never area — it was self-inflicted boundary damage.** Tail
   decomposition (production vs anchored-seed): **bbox ratio 1.000, hpwl and area
   identical**; the whole gap is `v_rel` (idx95: boundary 0→**21** of 37 tagged
   blocks). The `anchored` re-ASAP scrambled boundary-tagged blocks off their
   walls; the projection only partially restored them. This overturned the earlier
   "area-ratchet" hypothesis — item 2 (skeleton/area compaction) would have fixed
   nothing.
2. **Fixing the damage (legal-hint skeleton) removes the gap and the headroom
   together.** With the legal hint used directly as the skeleton, `anchored
   (production) == production` exactly on every tail case. But then the search has
   no downhill below production: production already matches on hpwl+area+violations
   and the now-near-identity realizer cannot exceed it. 86–99 accepted moves per
   case (v2) moved `best` by exactly 0.0000.
3. **The two anchored variants are a forced trade-off, and neither wins the
   tail.** Re-ASAP-always = search downhill (A3 17%) but a damaged seed (search
   only claws back its own damage); legal-hint-preserve = clean seed but no
   downhill. Root truth: **local (order,shape) perturbation + a preserve-or-
   compact realizer cannot exceed a strong SA optimum that is already quality-
   tight.** The +0.1 "gaps" were artifacts, not opportunity.
4. **The `faithful` ratchet (P1) is the same fact from the other side:** the hint
   floor that gives perfect fidelity forbids compaction (0/185 bbox shrink),
   improving moves are noise (best −0.0002). Preserve ⇒ no search; compact ⇒
   damage. There is no realizer setting in this family that both preserves a good
   seed and offers sub-production downhill on the tail.

## Keepers

- **`faithful` realizer** — exact fidelity for an already-optimal external hint
  (round-trip tax +0.000). Use when you have a good layout and want to score it,
  not to improve it.
- **`anchored` realizer (legal-hint skeleton)** — non-destructive on legal hints
  (`anchored(production) == production`), compaction-capable on illegal ones
  (analytic 9.95→~5.8, shelf→0). Portfolio-safe polish: ties production, wins a
  couple of small (n<100) cases; extrapolated full-100 portfolio(min) ≈ 1.272
  (delta ≈ −0.0003). **Banking it needs a fresh production rerun** (cache is old
  config). Do not treat as a promotion.
- **The four probes** — reusable instruments for any future decoder/realizer work.

## Do not relitigate

- **Beating production on the n≥100 tail via (order,shape) local search or
  analytic hints through any of these realizers.** Disproved across P1/P2/A3/pilot
  v1+v2 + the multi-seed confirmation. The binding cause is *no tail headroom*, not
  budget or move-set tuning: 1200 moves × 3 seeds on test99/95/97/94 found no run
  below base.
- **Item 2 (skeleton/area compaction).** Mechanistically moot on the tail — bbox
  already matches production (ratio 1.000). It addresses a problem that does not
  exist there.
- **`faithful` as a search engine.** It is a ratchet by construction.

Any future headroom is **upstream in the production solver itself** (the thing
that sets the ceiling), not in realizer/decoder/search design — out of this probe
chain's scope.

## Artifacts

- Code: `scripts/probes/{gen_decoder_probe,analytic_hints,perturb_probe,anchored_sa_pilot}.py`
- Evidence JSON/logs (scratchpad): `prod_realize.*`, `A3_anchored.*`,
  `sa_pilot.*` (v1), `sa_pilot_v2.*`, `confirm.*`, `compact_regress.log`.
- Regression: `uv run pytest -q` → **381 passed, 1 skipped**; faithful **1.2724**
  and compact **1.6535** byte-identical throughout.
