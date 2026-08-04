# `PARTNER_ANYTIME_LADDER` — anytime direct-refine ladder (task #10)

Status: implemented, **default off**, no evaluator evidence yet (per-case
proxy + micro-run evidence only; the paired full-100 is the scheduler's).
Flag surface: `PARTNER_ANYTIME_LADDER`, `PARTNER_ANYTIME_SECURE_MIN`,
`PARTNER_ANYTIME_SECURE_MAX`, `PARTNER_ANYTIME_BUILD_MULT`,
`PARTNER_ANYTIME_SECURE`, `PARTNER_ANYTIME_TIGHTEN`.
Code: `partner/layout_refiner.py` (`refine_prediction` only).
Tests: `tests/test_partner_anytime_ladder.py` (30).
Prior art: `docs/design/2026-08-04-early-exit-true-time-reduction.md` (time
anatomy), `docs/experiments/2026-08-04-low-budget-frontier-pool-gate.md`
(the ladder hypothesis this note discharges).

---

## 0. The claim being attacked

Paired full-100 evidence (0804): the `PARTNER_NREF=15` direct-refine slots are
worth **−0.126 noRT at `BUDGET_MAX=3.5`** (ctrl35 1.130 vs ctrl35_dmoff
1.2563) and **≈ 0 at `BUDGET_MAX=2.0`** (pg2_rep2 1.2844 vs pg2_dmoff 1.2875);
re-balancing slots (NREF=6) does nothing. Hypothesis on record: "the first
rung needs ~3 s; 3.5 s finishes it, 2.0 s does not." This note measures the
failure, and it is worse than the hypothesis.

## 1. Failure-mode diagnosis

Method: single-process micro-runs (`scratchpad/anytime_probe.py`), real
validation cases, real Direct predictions sampled once on CPU and cached, each
stage of `refine_prediction` wrapped with a timer. Plus one real single-case
evaluator run with `REFINER_DEBUG=1` as the production cross-check.

### 1.1 It is not (a) "legalization does not finish" alone — it is all four

Trace, case 95 (n=116), worker span 1.7 s (the `MAX=2.0` tier):

```
t=0.000 +0.171  build        (_Refiner for rung 0)
t=0.171 +0.363  lsoft fine   rung 0 FIXED frame -> FAILED (ovl 9..15)
t=0.535 +0.132  build
t=0.667 +0.373  lsoft        rung expand=0.02 -> FAILED   (deadline=None!)
t=1.041 +0.132  build
t=1.173 +0.471  lsoft        rung expand=0.28 -> LEGAL
t=1.643 +0.202  lsoft        tag recovery
t=1.845 +0.000  tighten      <- 0 s: the bbox recovery never runs
t=1.853 +0.107  build        (r2)
t=1.960 +0.000  run          <- 0 s: the HPWL refiner never runs
t=1.961 +0.001  seats        steps 6 / 6.5 / 7 all self-disabled
elapsed 1.96 s on a 1.70 s deadline  (+0.26 s OVERRUN)
```

The other two predictions never legalize at all → `refine_prediction` returns
`None` → the pool slot delivers nothing.

Four distinct defects, each independently sufficient:

1. **(a) all-or-nothing delivery.** The candidate exists only if some rung
   legalizes before the ladder deadline. 2/3 slots returned `None` at 1.7 s;
   at 1.2 s, 3/3.
2. **(b) tail starvation.** When a rung does legalize, it is the *loose* one
   (0.28 frame expansion ⇒ bbox 1.65 × `area_ref`), and the two stages that
   recover that inflation — `_tighten` (frame anneal) and `_Refiner.run`
   (squeeze + deflate + HPWL) — get **0.000 s**. The delivered candidate
   scores 1.53 on the pool's own proxy vs ~1.14 for a column restart: legal,
   delivered, and unusable.
3. **(c) ordering.** The span is spent on the two rungs that fail (0.36–0.56 s
   and 0.37–0.80 s) before the rung that succeeds (0.47 s).
4. **(d) unbounded calls.** The ladder rungs call `legalize_soft()` with
   `deadline=None`. On synthetic instances at a 0.5 s span the shipped path
   ran **0.63–1.47 s** — up to 3× the deadline. In the pool the case ends when
   the *slowest* worker returns, so this is raw runtime, and it also eats the
   `_parallel_solve` margin (`deadline_A − min(0.30, …)`).

Also structural: the tail gates are **absolute** (`t_hard − 0.4 / − 0.9 /
− 2.5`) while the carved reserve is `res = 0.3 · slice` = **0.51 s** at a 1.7 s
span. `t_hard − 0.9` is in the past before the reserve begins, so violation
repair, `_lock_compact` and the seats self-disable: the reserve is carved off
the search and then thrown away.

### 1.2 Production cross-check — the channel is dead on big cases even at 3.5 s

One real evaluator case (`test_id=95`, n=116) under the promoted env
(`MAX=3.5`, pool on, `NREF=15`, `PRESCREEN_V`, `OVERSAMPLE=4`):

* 15/15 refine workers print `[rp] rung FIXED failed ovl=59..80` (under
  31-way pool contention each worker gets far less CPU than the
  single-threaded probe, which reached ovl 9–15);
* **no** `[rp] rung ... legal` line at all;
* `[psel]` lists three `col` candidates and **zero `dir` candidates**.

So on the largest band the 15 slots are already 100 % waste at the promoted
budget — the low-budget tiers only make visible what is already true at n≳110.

### 1.3 Cost / value per stage (n=116, single thread)

| stage | cost | what it buys | value per second |
|---|---|---|---|
| `_edge_seat` + `_cluster_seat` | **~1 ms** | V −2…−3, monotone (kept only if V strictly drops) | enormous |
| secure rung (0.28, no pins) | build 0.13 + 0.47 s | `None` → a legal candidate | the whole slot |
| `_tighten` (frame anneal) | ~0.7 s | bbox 1.65 → 1.19 × `area_ref` (≈ −0.23 proxy) | ~0.33 /s |
| `_Refiner.run` | ~1.0 s | HPWL −5…10 % (≈ −0.05 proxy) | ~0.05 /s |
| rung 0 (fixed frame) | 0.5 s | best candidate when it works; **0/15 at n≈116** | n-dependent |
| tight rungs 0.02–0.18 | 0.4–0.8 s each | ~0 at n ≳ 110 | ~0 |
| step-6 repair `_attempt` | 0.2–0.6 s | occasionally V −1, frequently fails | low |
| step-7 recompress | ≥ 2.5 s (full rerun) | real, but only on long budgets | long-budget only |

`_Refiner.__init__` is O(n²) and costs 0.13 s at n=116 / 0.019 s at n=100 on
this machine — 3–4 builds per pipeline. That makes it the natural **cost
unit** for self-scaling budgets (§2).

## 2. What the flag does

Design rule: *never reorder what already works; bound what is unbounded, and
guarantee the delivery path out of a reserve sized in the instance's own cost
unit.*

1. **Reserve, in build units.** After timing the first `_Refiner` build
   (`t_build`), the ladder keeps
   `keep = clamp(BUILD_MULT · t_build, SECURE_MIN · span, SECURE_MAX · span)`
   (defaults 6.0 / 0.25 / 0.65) back from the tight rungs; the tight phase
   ends at `t_tight = deadline − keep`. Because `1 − SECURE_MAX = 0.35` is
   exactly the fixed-frame rung's own shipped share, **ANYTIME never shortens
   rung 0's window** — only the intermediate rungs and the salvage branches.
   No absolute seconds anywhere: the same rule yields 0.78 s at n=116/1.7 s
   and 0.14 s at n=100/0.8 s.
2. **Every legalization is bounded.** Tight rungs get `min(deadline,
   t_tight)`; the secure rung gets `SECURE · remaining` (0.55). The
   `deadline=None` calls are gone, which is where the 3× overrun lived.
3. **A secure rung is always reachable.** `rung_cap = min(rung_cap, t_tight)`
   makes the existing `continue` jump straight to the 0.28 rung once the tight
   window closes, and an **escape rung (0.50, no pins)** is appended: reached
   only when 0.28 also failed and time is left. A loose candidate loses
   selection at worst; a missing candidate wastes the slot outright.
4. **The quality tail is funded.** `_tighten` gets `TIGHTEN · remaining`
   (0.45) instead of the absolute `now + 3.0` (which below a ~4 s deadline is
   either "everything" or, after a starved ladder, ≤ 0), and `run` takes the
   rest.
5. **Tail gates become reserve fractions.** `_tg(shipped_const, frac)` returns
   `min(shipped_const, frac · res)`, so repair / `_lock_compact` / seats
   actually run inside a 0.5 s reserve — never a larger budget than the
   shipped constant.
6. **Value-ordered tail.** A seat pass (`_edge_seat` → `_cluster_seat`, ~1 ms,
   monotone) runs **before** the step-6 repair attempts. Measured: without it
   a failed repair attempt consumed the reserve and cost a V −2 the seats
   would have banked (case 64 pred 0 at 1.7 s: +0.10 proxy → 0.000 after).
7. **Step 7 (recompression) is deliberately untouched.** It reruns the whole
   pipeline (3+ builds); a reserve-sized share of a 1.7 s deadline cannot fund
   it, and the absolute `t_hard − 2.5` gate self-disables at exactly those
   tiers.

Off path: `_any` is `False`, `t_tight = deadline`, `_rdl = None`, `_tg`
returns the shipped constant, the escape rung is not appended, and the early
seat pass does not exist — every added expression reduces to the pre-port
form. No added predicate draws randomness (asserted in the test file).

**Off-path fidelity, measured** (`scratchpad/anytime_bitexact.py`): the
pristine module and the patched module, flag off, on spans long enough that
every stage converges (so the pristine module is self-reproducible — verified
by running it twice) produce **byte-identical** layouts:

```
n=40 seed=0 span=8.0 : old=2b4df707a2c9313e old2=2b4df707a2c9313e new=2b4df707a2c9313e
n=40 seed=1 span=8.0 : old=66c9507ce4085291 old2=66c9507ce4085291 new=66c9507ce4085291
n=70 seed=0 span=10.0: old=7b13d2f5fe4e805c old2=7b13d2f5fe4e805c new=7b13d2f5fe4e805c
```

(The test file additionally asserts the *shipped deadline arguments*: with the
flag off some `legalize_soft` call must still be unbounded, `_tighten` must
still be handed `min(ladder_deadline, now + 3.0)`, and `_cluster_seat` must
still see `t_hard − 0.9` / `t_hard − 0.05`.)

## 3. Measured effect (per-case proxy, single thread)

Proxy = the shape `_parallel_solve.score` uses
(`(1 + 0.5((hp−hp_ref)/hp_ref + max(0, area/area_ref − 1))) · e^{2V/n_soft}`),
4 cached Direct predictions per case; best-of-slots is what the pool selection
would see. Harness: `scratchpad/anytime_ab.py`.

| case | span | off best-of | on best-of | Δ | slots delivered off → on |
|---|---|---|---|---|---|
| t95, n=116 | 3.2 s | 1.1946 | **1.1831** | −0.011 | 3/4 → 4/4 |
| t95, n=116 | 1.7 s | 1.5353 | **1.1831** | −0.352 | 1/4 → 4/4 |
| t95, n=116 | 0.8 s | none | **1.2326** | — | 0/4 → 2/4 |
| t79, n=100 | 3.2 s | 1.1623 | 1.1623 | 0.000 | 4/4 → 4/4 |
| t79, n=100 | 1.7 s | 1.4377 | **1.3774** | −0.060 | 4/4 → 4/4 |
| t79, n=100 | 0.8 s | 1.3695 | 1.4603 | **+0.091** | 1/4 → 1/4 |
| t64, n=85 | 3.2 s | 1.2359 | 1.2359 | 0.000 | 4/4 → 4/4 |
| t64, n=85 | 1.7 s | 1.2448 | 1.2359 | −0.009 | 4/4 → 4/4 |
| t64, n=85 | 0.8 s | 1.3239 | 1.3239 | 0.000 | 3/4 → 4/4 |

Synthetic instances (`scratchpad/anytime_synth.py`, n ∈ {40, 70, 100} × 2
seeds) — delivery and wall clock:

| span | off delivered | off elapsed | on delivered | on elapsed |
|---|---|---|---|---|
| 0.5 s | 0/6 | 0.63–1.47 s (**up to 3× the deadline**) | 2/6 | 0.34–0.50 s |
| 1.0 s | 0/6 | 0.71–1.60 s | 6/6 | 0.72–1.01 s |
| 2.0 s | 3/6 | 1.44–1.86 s | 6/6 | ≤ 1.99 s |
| 4.0 s | 6/6 | ≤ 3.44 s | 6/6 | ≤ 3.62 s |

## 4. Expected recovery per tier, and what could go wrong

Ranges, because the proxy is not the evaluator and the pool turns per-slot
effects into a max-of-15 order statistic:

| tier (`BUDGET_MAX`) | worker span | expectation | reasoning |
|---|---|---|---|
| 3.5 (promoted) | ~3.2 s | **−0.01 … 0.00** | the channel is dead only at n≳110 today; 3/4→4/4 slots and −0.011 there, exactly neutral at n≤100 |
| 2.5 | ~2.1 s | −0.02 … −0.05 | interpolation between the 3.2 s and 1.7 s columns |
| 2.0 (**the gate**) | ~1.7 s | **−0.03 … −0.08** | the channel goes from ~0 usable slots to 4/4 on the tail band and −0.060 at n=100; the ladder is worth −0.126 at 3.5 s, so recovering half is the central case |
| 1.0 | ~0.8 s | −0.02 … +0.01 | mixed: +0.091 on one case, 0.000 on another, delivery where nothing delivered |
| 0.5 | ~0.4 s | −0.01 … −0.03 | off delivers essentially nothing here; also the tier where the overrun fix matters most |

Independently of noRT: **raw runtime should improve at every tier below ~2.5 s**
(the shipped path overruns the worker deadline by 0.26 s at n=116/1.7 s and by
up to 3× on 0.5 s spans; ANYTIME stays inside it). That is a budget-layer
input, not a quality claim.

Risks, in descending order:

1. **Insurance can tip a barely-succeeding fixed rung into failure.** Measured
   once (t79 n=100 at 0.8 s, +0.091 on the surviving slot): the reserve took
   the salvage branch's time. Bounded by `SECURE_MAX = 0.65` (rung 0's own
   window is never cut), so the residual exposure is the ≤6-overlap salvage.
   Next tuning lever if the A/B shows it: extend the salvage past `t_tight`
   while `_overlap_count() ≤ 6` — a reusable instance signal, not a clock.
2. **A more permissive tail can spend the reserve on failing repair
   attempts.** Mitigated by the early seat pass (§2.6); re-verify if the
   repair profile ever changes.
3. **Loose / escape-rung candidates are legal but wide** (bbox up to
   1.65 × `area_ref`). They lose the 1.5 %-margin selection, so they cost
   nothing but the slot they already occupy; today's `None` contributes
   nothing either. The only asymmetric case is "every column restart failed",
   where a wide legal layout beats the row fallback.
4. **Evidence class.** All of the above is single-threaded proxy evidence on
   3 real cases + 6 synthetic ones. Pool contention makes each worker slower,
   which strengthens the case directionally (production reaches only ovl
   59–80 on rung 0) but the magnitude does not transfer linearly.

## 5. Gate

Paired, contemporaneous, ≥ 2 reps, `100/100 feasible` as a precondition:

1. **Arm A (the target):** `MAX=2.0`, `PARTNER_POOL_GATE=0`, direct on,
   `+PARTNER_ANYTIME_LADDER=1` vs the same without. Promote the low-budget
   regime on **Δ ≤ −0.01** against the 1.284–1.288 family.
2. **Arm B (no-regression):** promoted env (`MAX=3.5`) ± the flag. Requirement:
   no regression (|Δ| ≤ the ~0.002 control σ), and report Δ raw runtime — the
   overrun fix should show up as a shorter tail.
3. Only if A and B pass: tier sweep `MAX ∈ {2.5, 1.0, 0.5}` to re-price the
   low-budget frontier (current marks 1.2884 / 1.3044 / 1.3515).
4. Knob sweep (second order, only once a direction exists):
   `PARTNER_ANYTIME_SECURE_MIN ∈ {0.20, 0.25, 0.35}`,
   `PARTNER_ANYTIME_TIGHTEN ∈ {0.30, 0.45, 0.60}`.

Interop verified by construction and by test: `PARTNER_EARLY_EXIT` (its
`t_hard` clawback shrinks the reserve, which the fractional gates follow),
`PARTNER_REFINE_STALL_STOP`, and `PARTNER_SA_KERNEL=numba` (no dispatch code
is touched; smoke run delivers a legal layout with both flags on).
