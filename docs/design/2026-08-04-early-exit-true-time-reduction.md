# `PARTNER_EARLY_EXIT` — true per-case time reduction (task #9)

Status: implemented, **default off**, no evaluator evidence yet.
Flag surface: `PARTNER_EARLY_EXIT`, `PARTNER_EARLY_EXIT_WINDOW`,
`PARTNER_EARLY_EXIT_MIN_WINDOW`, `PARTNER_EARLY_EXIT_DEBUG`.
Code: `partner/column_sa_legalizer.py`, `partner/layout_refiner.py`,
`partner/contest_optimizer.py`. Tests: `tests/test_partner_early_exit.py`.

---

## 1. Time anatomy of one case

`_time_budget` (`partner/contest_optimizer.py` L86) sets
`budget = clamp(0.06·e^(n/20), BUDGET_MIN, BUDGET_MAX)` and every loop below
is `while time.time() < deadline`. The fork is *deadline-bounded*: official
runtime **is** the budget. Nothing sleeps — the whole chain is
`while`-loops against absolute deadlines plus two `map_async(...).get(timeout)`
joins.

### 1.1 The chain (production env: `BUDGET_MAX=3.5`, `VKILL_OFF=1`)

```
solve()  t=0
  budget = clamp(0.06 e^(n/20), 0.8, 3.5);  deadline = t0 + budget
  [vkill carve — dead under VKILL_OFF=1]
  ── SERIAL HEAD (t_pre) ──────────────────────────────────────────────
  tensor .cpu() marshalling
  _heuristic_init(...)
  _diffusion_refine(...)              GPU DDIM seed refine (steps=4)
  _ColumnOptimizer(opt1) construction  _resolve_shapes/_build_hpwl_arrays/
                                       _choose_frame/_build_units
  ── legalize_rectangles(deadline) ────────────────────────────────────
  budget > 3.0 and pool ready ?
    YES → _parallel_solve                      (n ≳ 79)
          payload marshalling (24 × numpy copies)
          map_async(_worker_solve, 9..16 payloads)
          sample_fn(n_ref)  ← BLOCKING GPU sampling in the parent
          map_async(_worker_refine, n_ref payloads)
          res.get(); ref_res.get()   ← waits for the SLOWEST worker
          score / select / _ensure_no_overlap
    NO  → opt1.run()                           (n ≲ 78, single-threaded)
          prepare → probe(3 chains) → finish()
  ── TAIL (t_post) ────────────────────────────────────────────────────
  _pick_best (pool path: empty), _violation_kill (off)
```

**Finding 1 — the path splits at n ≈ 78.** `legalize_rectangles` only enters
the pool when `budget > 3.0`, i.e. `0.06·e^(n/20) > 3.0` ⟺ `n > 78.2`. At the
promoted `BUDGET_MAX=3.5` roughly the lower half of the validation set never
touches the restart pool at all (and, as a corollary, never consumes the
Direct/flow channel — `sample_fn` is constructed in `solve` but only *used*
inside `_parallel_solve`). Small/mid cases are one single-threaded chain;
only the tail is a 24-worker fan-out.

### 1.2 Who idles to the deadline

Every stage is handed an **absolute** deadline, so a stage that converges
early does not shorten anything — it silently donates its remaining share to
the next stage in the same worker. The donation chain:

| stage | bounded by | convergent on its own? | where the saving went |
|---|---|---|---|
| `probe` (3 chains) | `t_each` each | no (SA runs to time) | → `finish` |
| `_anneal` in `finish` | `t_end` | no | → polish + refiner |
| `_greedy_polish` | `deadline − refine_t` | **yes** (`while improved`) | → refiner |
| `refine_positions` → `_Refiner.run` | `deadline − 0.05` | only with `REFINE_STALL_STOP` | → nothing (worker ends) |
| `refine_prediction` rungs | `rung_cap` | yes (first legal rung wins) | → `r2.run` |
| `r2.run` (`_worker_refine`) | `deadline − 0.02` | only with `REFINE_STALL_STOP` | → step 7 recompression rerun |
| step 7 recompression | `t_hard − 2.5` gate | quality-triggered | → nothing |
| case | `max` over 24 workers | — | — |

This is exactly why the two existing flags behaved the way they did:

* **`PARTNER_SA_STALL_STOP` measured quality-negative** (+0.0016 noRT) — the
  reclaimed anneal time was re-spent by the polish and the stage-2 refiner
  *in the same worker*, and the worker still ran to `worker_deadline`. Pure
  loss: less annealing, same wall clock.
* **`PARTNER_REFINE_STALL_STOP` promoted** (−0.0049 noRT) — `_Refiner.run` is
  the terminal sink of both worker classes, so its reclaimed time flowed into
  `refine_prediction`'s later passes (notably the step-7 recompression rerun),
  which are worth more per second than a converged refiner sweep.

`PARTNER_EARLY_EXIT` inverts the second one on purpose: **return** the time
instead of re-spending it.

### 1.3 Fixed overhead (unverified — instrument before trusting)

`t_pre` (serial head) and `t_post` (selection tail) are **not** reducible by
early exit: they run before the first worker is dispatched / after the last
one returns. Their size decides the floor of the whole exercise, and I did
not measure them (no multi-case runs allowed in this worktree — a paired
sweep chain was live on the host). Order-of-magnitude expectation for n≈100:
tens of ms for the parse/`_ColumnOptimizer` build, plus whatever the GPU
seed-diffusion (`_diffusion_refine`, DDIM `steps=4`) costs. **This is the
first thing to measure**, via the instrumentation added for it:

```bash
PARTNER_EARLY_EXIT_DEBUG=1 ...   # one line per case on stderr:
# [ee] n=104 budget=3.50 pre=0.083 solve=3.402 post=0.006 total=3.491
```

Only the `solve=` term is addressable. If `pre` turns out to be ≫ 0.1 s, the
0.2 s/case endgame is a *seed-path* problem, not a search problem, and this
flag cannot get there alone.

---

## 2. Flag semantics

`PARTNER_EARLY_EXIT=1` means: **every stage keeps its planned share and
returns the rest to the case.** Concretely it does two things.

**(a) It implies both stall detectors.** The convergence tests themselves are
the existing, already-reviewed ones — `_anneal`'s `stall_ref_cost/stall_ref_time`
window and `_Refiner.run`'s `ref_key/ref_time` window. EARLY_EXIT turns both
on; explicit `PARTNER_SA_STALL_STOP=0` cannot turn them back off (the stop is
what makes the exit possible), but the per-phase window/eps overrides still
apply. Window precedence:

1. `PARTNER_SA_STALL_WINDOW` / `PARTNER_REFINE_STALL_WINDOW` when set,
2. `PARTNER_EARLY_EXIT_WINDOW` (default **0.15**),
3. the historical stall-stop default 0.25.

The EARLY_EXIT default is *tighter* than 0.25 because the objective changed:
under the stall stops the detection window was time that got re-spent, so its
length was free; here every window second is wall clock not returned. The
window stays a **fraction of the phase's own span**, so it self-scales to
`PARTNER_BUDGET_MAX` with no retuning — a 0.5 s/case tier gets a 0.5 s-scaled
window. `PARTNER_EARLY_EXIT_MIN_WINDOW` (default 0.05 s) floors it so a tiny
budget cannot stop a phase before its first productive round.

**(b) Clawback.** When a stage returns before its planned end, the caller
shrinks its own deadline by the unspent amount:

* `probe` → `run` / `legalize_rectangles`: `_probe_unspent` is subtracted
  from the deadline handed to `finish`.
* `_anneal` chains → `finish`: `saved` accumulates per run (so with `runs=2`
  the second chain does not absorb the first chain's saving) and is
  subtracted from `deadline` before the polish/refine boundaries are used.
* `_greedy_polish` → `finish`: its unspent tail (it was *already*
  convergence-bounded) is subtracted before `refine_positions`.
* `r2.run` → `refine_prediction`: `t_hard` is shrunk by the unspent share, so
  steps 6 / 6.5 / 7 keep their own carved reserve and nothing more, and the
  step-7 `t_hard − 2.5` gate self-disables when the recompression rerun no
  longer fits inside the *planned* span.
* `legalize_rectangles` → `solve`: nothing to do — `map_async(...).get()`
  already returns the instant the last worker returns.
* `_violation_kill`: `t_kill` was absolute, so a legalizer that returned early
  would have handed vkill the entire reclaimed tail; under EARLY_EXIT it is
  capped at the carved reserve. (Dead today under `VKILL_OFF=1`.)

Note what clawback deliberately does **not** do: it never cancels
quality-triggered work. Step 6 (violation repair, gated on `V0 > 0`) and step
7 (recompression, gated on `bbox > 1.08·area_ref`) still run inside their
carved reserve. EARLY_EXIT returns *unspent* time, not *planned* time.

**(c) One non-obvious sign fix.** `_Refiner.run`'s discrete phase re-scales
its stall window to `frac × (deadline − now)`. That is right when the saving
is re-spent, and exactly wrong here: a continuous phase that converged after
0.2 s of a 3 s slice leaves `deadline − now ≈ 3 s`, i.e. it hands the discrete
phase a nearly full-size window — the very wall clock the flag exists to
return. Under EARLY_EXIT the discrete window is sized off the continuous
phase's own span (`phase1_span`), capped by the historical value so it is
never longer than today's.

**Off path**: `_early_exit` is `False`, `_stall_frac` stays `0.0`, `saved` /
`_probe_unspent` stay `0.0`, and every added expression reduces to the
pre-port form. None of the added predicates draw randomness, so the rng
stream is untouched (asserted in
`test_early_exit_consumes_no_extra_randomness`).

---

## 3. Where the saving comes from, per tier

Let `W` = window fraction (0.15), and note that a converged phase still burns
one window before it breaks. So the floor of a stage is
`t_converge + W·span`, and with `k` stall-detecting phases in series the
theoretical best case is roughly `t_pre + Σ W·span_i + t_post`.

**Small / mid cases (n ≲ 78, sequential, budget 0.8–3.0 s).** One chain, no
straggler problem. Stall-detecting phases: probe (×3), the `finish` anneal,
the refiner's two phases. Best case ≈ `t_pre + ~0.15·budget`. Realistic:
**35–60 % of budget returned on cases that genuinely converge, ~0 % on cases
that do not.** These cases are cheap in absolute seconds but numerous.

**Tail cases (n ≳ 79, 24 workers, budget 3.05–3.5 s).** The case ends when
the **slowest of ~24 workers** returns. A max over 24 independent SA
convergence times sits far into the upper tail, so the expected saving is much
smaller than the per-worker average saving — plausibly **10–25 %**. This is
the honest pessimistic read, and it is the band that dominates raw runtime.
Cutting stragglers (returning the results that are in and abandoning the rest)
would fix it, but it changes the candidate set and therefore the layout —
deliberately **not** in this flag.

**`BUDGET_MAX = 1.0`.** Everything is sequential (`budget ≤ 3.0`), budgets sit
at the `BUDGET_MIN = 0.8` floor for most of the set. Windows scale down with
the span, the 0.05 s floor starts to bind on the shortest phases. Expected
**20–40 %**.

**`BUDGET_MAX = 0.5` — blocked by a different knob.** `_time_budget` computes
`max(BUDGET_MIN, min(BUDGET_MAX, b))`, and `BUDGET_MIN` defaults to **0.8**,
so `PARTNER_BUDGET_MAX=0.5` on its own changes *nothing* — the floor wins.
Reaching the 0.2 s/case endgame requires lowering `PARTNER_BUDGET_MIN`
alongside (e.g. `PARTNER_BUDGET_MIN=0.15 PARTNER_BUDGET_MAX=0.5`), and at that
tier the binding constraint becomes `t_pre` (serial head, §1.3), not the
search. Measure `pre=` with `PARTNER_EARLY_EXIT_DEBUG=1` before scheduling
that experiment.

Residual idle-to-deadline consumers **not** addressed (both default off, both
size their work off the absolute case deadline, so they would re-spend the
reclaimed time): `PARTNER_PHASE_B` and `PARTNER_FUSION` in `_parallel_solve`.
If either is ever promoted, it needs the same clawback treatment.

---

## 4. Risk and the gate

**This change strictly removes search.** Every returned second is a second of
annealing/refinement not done, so the expected no-runtime delta is ≥ 0
(worse or equal), never better. The prior evidence quantifies it:

* `PARTNER_SA_STALL_STOP` alone cost **+0.0016** noRT *while still re-spending*
  the reclaimed time. EARLY_EXIT also removes the re-spend, so its SA-side
  cost should be ≥ that.
* `PARTNER_REFINE_STALL_STOP` *gained* **−0.0049** noRT precisely by
  re-spending. EARLY_EXIT cancels that gain.

So the naive prediction for "promoted config + `PARTNER_EARLY_EXIT=1`" is
roughly **+0.005 … +0.008 noRT**, which **breaches the ≤ +0.003 gate on its
own**. That is not a reason to shelve it — it is a reason to A/B it correctly:

> **The right comparison is iso-runtime, not iso-budget.** EARLY_EXIT is not a
> drop-in quality change; it is a *runtime buyer*. The decision experiment is
> `control` vs `EARLY_EXIT + PARTNER_BUDGET_MAX raised until the measured mean
> per-case runtime matches control`. If that arm wins (or ties) on noRT, the
> flag converts converged-case idle time into extra search on the cases that
> actually need it, and the projected (alpha-weighted) score improves.
> Judging it at fixed `BUDGET_MAX` measures only the loss half of the trade.

Gating plan (for the scheduler, in a clean environment):

1. **Anatomy first, cheap.** One full-100 run with
   `PARTNER_EARLY_EXIT_DEBUG=1`, flag **off**, to get the `pre/solve/post`
   split and the raw runtime tail. This alone decides whether the 0.2 s target
   is reachable at all.
2. **Arm A (loss measurement):** promoted env `+PARTNER_EARLY_EXIT=1`, same
   `BUDGET_MAX=3.5`. Records ΔnoRT (expected positive) **and** Δruntime
   (expected clearly negative) — the exchange rate.
3. **Arm B (iso-runtime):** `PARTNER_EARLY_EXIT=1` with `BUDGET_MAX` raised so
   mean runtime ≈ control. Promote only if ΔnoRT ≤ 0 and Δprojected < 0.
4. Window sweep `PARTNER_EARLY_EXIT_WINDOW ∈ {0.10, 0.15, 0.25}` only after
   arm A/B pick a direction — it trades detection latency against premature
   stops and is a second-order knob.

Contemporaneous paired rounds as usual (≥ 2 reps; only same-round deltas are
trusted), `100/100 feasible` is a hard precondition for reading any number.

Hard-legality risk is low by construction: an early break returns the `best`
snapshot, which is a layout the phase already accepted, and every exit path
still goes through `_ensure_no_overlap` / the existing repair steps. The
invariant tests in `tests/test_partner_early_exit.py` assert no-overlap,
exact areas and unchanged block count on early-exit returns.
