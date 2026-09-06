# `PARTNER_REFINE_KERNEL` — flat-array numba kernel for the direct-refine rung

Status: implemented, **default off**, no evaluator evidence yet (single-thread
micro-benchmark + bit-exactness evidence only; the paired full-100 is the
scheduler's).
Flag surface: `PARTNER_REFINE_KERNEL=numba`, `PARTNER_REFINE_KERNEL_WARMUP`,
`PARTNER_REFINE_FASTBUILD`.
Code: `src/solver/refine_numeric_kernel.py`, four dispatch sites in
`src/solver/layout_refiner.py`.
Tests: `tests/test_partner_refine_kernel.py`.
Prior art: `src/solver/sa_numeric_kernel.py` (`PARTNER_SA_KERNEL`, the template),
`docs/design/2026-08-04-anytime-ladder.md` (the failure this unblocks),
`docs/design/2026-08-04-early-exit-true-time-reduction.md` (time anatomy).

---

## 0. What this is for

Two verdicts on record blame the same thing — the rung is too slow:

1. At the promoted `BUDGET_MAX=3.5`, on the `n ≳ 110` band, **15/15 refine
   workers print `[rp] rung FIXED failed ovl=59..80` and no `dir` candidate
   ever reaches `[psel]`** (anytime-ladder note §1.2). The 15 direct-refine
   slots are 100 % waste there, so the ladder's measured −0.126 noRT never
   extends to the largest instances.
2. `PARTNER_ANYTIME_LADDER` was a wash / small loss in the paired A/B. Its
   mechanism (anytime re-ordering + bounded rungs) is real in a single-thread
   proxy — n=116 @1.7 s went 1.535 → 1.183 — but under 31-way pool contention
   the rung body is simply too slow for re-ordering to save.

Both dissolve if a rung finishes ~2.5× sooner. This flag does that.

## 1. Hot-spot anatomy

Method: single thread, real validation cases, real Direct predictions sampled
once and cached, `legalize_soft()` replayed with a per-section timer
(`scratchpad/refine_profile.py`, `scratchpad/refine_axis_split.py`). A rung is
`legalize_soft`, which is `legalize()` four times over (the 0.88 / 0.94 / 0.97
/ 1.0 regrow ladder); `legalize()` is `_axis_pass` ×2 per sweep plus eviction.

Aggregated over the six n=116 rungs measured (3 predictions × {fixed frame,
0.28 frame}, 2.418 s total):

| section | what it computes | share |
|---|---|---|
| `_evict` | anchor × variant scan for a free landing slot | **35.4 %** |
| `_axis_constraints` **B** | pair → `tight` dict → `edges_in/out` Python loop | **24.8 %** |
| `_axis_constraints` **C** | per-group frame bounds (`c0[mem].min()` ×2 ×G) | 13.6 % |
| `_axis_pass` **E** | forward assignment + `_move` | 8.6 % |
| `_axis_constraints` **A** | the O(n²) numpy separation masks | 7.7 % |
| `_axis_constraints` **D** | backward longest path over `edges_out` | 5.6 % |
| `_has_overlap` | O(n²) numpy overlap test | 1.7 % |
| everything else | `_reshape_chain`, `_free_rects`, `_overlap_pairs`, … | 2.6 % |

Per-call costs at n=116: `_axis_constraints` 550–620 µs, of which the numpy
part is only ~185 µs — the rest is Python-level loops over ~1000 pairs, ~660
group-pair edges and 89 groups. `_evict` is ~1.06 ms, of which ~90 % is the
anchor scan (≤40 anchors × 4 variants × 6 numpy ops on length-n arrays; each
op is ~6 µs of numpy dispatch on 116 doubles).

**Nothing here is an object-graph search.** `_Group` and `_AxisCons` are
containers for numbers; the algorithm is arithmetic. So the same move that
worked for the column-SA layout pass applies: re-express it over flat arrays.

**Amdahl.** Porting A–E + `_has_overlap` but leaving `_evict` in Python caps
the rung at 1/(0.354+0.026) = **2.63×** — it clears the ÷2.5 bar only barely,
and not at all on the fixed-frame rung where `_evict` alone is 62 %. With
`_evict`'s scan ported too the serial residual is ~3 %, i.e. the rung is not
serial-limited; the realised number is set by kernel quality. That decided the
scope.

## 2. What is ported, and what is not

Ported (`src/solver/refine_numeric_kernel.py`):

* `_axis_pass` **with `hold=True` only** — sections A+B+C+D+E fused into one
  `njit` call. This is the legalization sweep (`legalize`, `_legal_check`),
  i.e. the entire rung. The `_AxisCons` object is gone from this path.
* the `_evict` anchor × variant scan (the Python prologue stays).
* `_has_overlap`, `_overlap_count`.

Deliberately not ported:

* **`_axis_pass(hold=False)`** — the HPWL median sweep in `run`. Its per-group
  target is `_median_shift` → `_wmedian` → `np.argsort(vals)` with numpy's
  *default* (introsort) kind. The tie permutation is not reproducible from the
  algorithm alone, and `_wmedian` then `cumsum`s the reordered weights, so a
  different tie permutation is a different float sum. Porting it would turn
  this toggle from a pure speed change into a search-trajectory change.
* **`_optimal_point`** — same reason; hence `_evict` keeps its prologue and the
  kernel takes the anchor order as an argument.
* `_reshape_chain` / `_relocate_to_free` / `_band_restack` and friends: object
  -graph search, < 1 % each in the rung profile.

## 3. Bit-exactness contract

Transcription, not reimplementation. Two places would be where a transcription
silently drifts, so they are argued explicitly:

1. **Edge ordering.** `_axis_constraints` fills `edges_in` / `edges_out` by
   walking `tight.items()` — Python dict *first-encounter* order. The kernel
   reproduces it exactly: `tight_idx` maps `(gi, gj)` to a slot in an
   append-ordered edge list, and the CSR is built by a counting sort that is
   stable in that append order. (The order is in fact inert — every consumer
   reduces with `max`/`min` of exactly-representable doubles, never a sum —
   but reproducing it costs nothing and removes the argument.)
2. **`sorted()` inside `_move`.** The cluster contact re-snap sorts `g.cH` by
   `P[a, 0]` with Python's stable `sorted`, evaluating every key *before* any
   assignment. The kernel snapshots the keys, runs a stable insertion sort,
   then assigns.

`np.argsort(key_of, kind="stable")` is reproduced with numba's
`kind='mergesort'`: a stable sort's output permutation is uniquely determined
by its input, so the two agree by construction — and the test file asserts it
on tie-dense inputs rather than trusting the argument.

Everything else is a literal transcription: same tolerances (`SEP_TOL = 5e-7`,
the `9e-7` of the overlap tests), same comparison directions, same early
exits, same `min`/`max` nesting. There are no vectorised reductions where the
Python original accumulates sequentially, and no fast-math.

### Measured

`scratchpad/refine_kernel_probe.py` — 12 rungs on t95 (n=116), Python path vs
kernel path from the same start state, compared with `np.array_equal`:
**12/12 bit-exact.** `tests/test_partner_refine_kernel.py` asserts the same
`==` contract per primitive (axis pass incl. `invert`, `_evict` incl.
`max_anchors=n` and an explicit `target`, both overlap tests), per rung
(`fine` and coarse, fixed and expanded frame), for `_tighten`, and for the rng
stream (`rng.getstate()` identical after a rung).

End-to-end `refine_prediction` parity is asserted only where the *deadline is
inert* — at a span where the Python path is self-reproducible. That is not a
weaker claim, it is the only well-posed one: `refine_prediction` is
deadline-driven, so wherever the deadline still binds, the kernel legitimately
does **more** work in the same wall clock and the layouts differ. That is the
point of the flag.

## 4. The build became the bottleneck — `PARTNER_REFINE_FASTBUILD`

With the rung at ~13× the cost centre moved. `_Refiner.__init__` costs 0.107 s
at n=116 and the ladder runs 3–4 builds per pipeline, so after the kernel the
*build* is ~2.4× a whole rung. Profiling it: **97 % is `_build_edges`**, and
the reason is algorithmic, not numeric —

```python
for gi, g in enumerate(self.groups):          # G groups
    for a, b, w in zip(opt.eI, opt.eJ, opt.eW):   # ... times every edge
```

— an O(G·|E|) rescan (89 × ~1700 at n=116). Bucketing the edges by group in
one pass is O(|E| + |P| + G) and appends in the same edge order, so every
`g.eM/eJ/eW/pM/pX/pY/pW` array is element-for-element identical (asserted, per
group, per attribute, in the test file). This is a pure-Python change; it
needs no numba and it is on whenever the kernel is, but
`PARTNER_REFINE_FASTBUILD=0` disables it alone so the two mechanisms stay
separately attributable in an A/B.

| n | `_Refiner.__init__` shipped | + fastbuild | speedup |
|---|---|---|---|
| 85 | 0.0323 s | 0.0029 s | 11.1× |
| 100 | 0.0156 s | 0.0015 s | 10.4× |
| 116 | 0.1068 s | 0.0167 s | 6.4× |

Consequence for `PARTNER_ANYTIME_LADDER`: its reserve is sized in **build
units** (`BUILD_MULT · t_build`, default 6.0). Both `t_build` and the rung
just moved by roughly an order of magnitude, so the ladder's knobs
(`BUILD_MULT`, `SECURE_MIN`, `SECURE_MAX`, `TIGHTEN`) are re-priced by this
change and must be re-swept, not merely re-tested.

## 5. Measured effect

`scratchpad/refine_kernel_bench.py`, single thread, cached Direct predictions,
mean of 3 predictions per band, both flags on:

| n | stage | shipped | kernel | speedup |
|---|---|---|---|---|
| 85 | rung, fixed frame | 0.5100 s | 0.0265 s | **19.2×** |
| 85 | rung, 0.28 frame | 0.1900 s | 0.0142 s | 13.4× |
| 85 | `_tighten` | 0.2663 s | 0.0201 s | 13.3× |
| 100 | rung, fixed frame | 0.9897 s | 0.0436 s | **22.7×** |
| 100 | rung, 0.28 frame | 0.2214 s | 0.0161 s | 13.8× |
| 100 | `_tighten` | 0.2999 s | 0.0218 s | 13.8× |
| 116 | rung, fixed frame | 0.5352 s | 0.0347 s | **15.4×** |
| 116 | rung, 0.28 frame | 0.2698 s | 0.0214 s | 12.6× |
| 116 | `_tighten` | 0.3753 s | 0.0305 s | 12.3× |

The target was ÷2.5. The fixed-frame rung — the one that is 15/15 dead in
production — gains most, because `_evict` (62 % of it) is the section with the
worst numpy-dispatch-per-useful-flop ratio.

JIT cost, fresh process (`warmup_cost()` in the same script):

| | cold cache | warm cache |
|---|---|---|
| `warm_process()` | 3.34 s | 0.114 s |
| first `_Refiner` after it | 0.107 s | 0.107 s (= the un-accelerated build) |

`warm_process()` compiles every entry point on a 2-block dummy and is
idempotent (2 µs on the second call). **Call it in the parent before the
restart pool forks** — children inherit compiled code and pay nothing.
Without a parent-side warm, the first `_Refiner` in each worker pays ~0.17 s of
cache load, which on a 1.7 s worker span is a whole first rung. A cold machine
pays the 3.3 s compile once; that must not happen inside a timed run.

Off path: `PARTNER_REFINE_KERNEL` unset ⇒ `_Refiner._nk is None`,
`_fast_build is False`, each dispatch site is a single `is None` test and
`numba` is never imported. `scratchpad/refine_kernel_offpath.py` measures the
stronger claim — the pre-port module (`git show ad3b26e:src/solver/layout_refiner.py`)
and the patched module, flag off, on spans long enough that every stage
converges (so the pristine module is self-reproducible, verified by running it
twice) — **4/4 byte-identical**:

```
n=40 seed=0 span=20: old=dd8152b142864f3a old2=dd8152b142864f3a new=dd8152b142864f3a
n=40 seed=1 span=20: old=66c9507ce4085291 old2=66c9507ce4085291 new=66c9507ce4085291
n=70 seed=0 span=25: old=df653e7613d6c805 old2=df653e7613d6c805 new=df653e7613d6c805
n=70 seed=1 span=25: old=236c9b060021663c old2=236c9b060021663c new=236c9b060021663c
```

## 5b. Scope warning for the A/B

The flag accelerates **every** `_Refiner` in the process, not only the ladder
rungs: `refine_positions` (the stage-2 slice that terminates `_worker_solve`),
`_lock_compact` and `compact_to_locks` all go through the same `hold=True`
axis pass and the same overlap tests. So the paired run measures "refiner
throughput", not "refine-slot throughput" — expect column-restart workers to
move too.

## 6. Gate

This flag is a **speed** change with a bit-exact per-rung contract, so its own
promotion evidence is the rung time and the delivery rate, not a proxy score.
What it unblocks needs the usual paired evidence:

1. **Delivery, promoted tier.** `MAX=3.5`, promoted env, `REFINER_DEBUG=1` on
   the n≳110 band: today 15/15 workers print `rung FIXED failed`. The
   post-condition is `[psel]` listing at least one `dir` candidate.
2. **Arm A (no-regression).** Promoted env ± `PARTNER_REFINE_KERNEL=numba`,
   ≥2 reps, `100/100 feasible` as a precondition. Expected: Δ noRT ≤ 0 (the
   rung either delivers where it did not, or converges earlier); report Δ raw
   runtime.
3. **Re-price `PARTNER_ANYTIME_LADDER`.** Its A/B verdict (wash at 3.5,
   +0.012 at 2.0) was taken against a rung that could not finish. Re-run the
   ladder gate (`docs/design/2026-08-04-anytime-ladder.md` §5) *with* the
   kernel on in both arms.
4. Only then: the low-budget tier sweep, where a ~n× faster rung changes what
   a 0.4–1.7 s worker span can buy.

Interop: no dispatch code of `PARTNER_SA_KERNEL`, `PARTNER_EARLY_EXIT`,
`PARTNER_REFINE_STALL_STOP` or `PARTNER_ANYTIME_LADDER` is touched. The
kernel is orthogonal to all four — it changes how long a stage takes, never
which stage runs.

### Compile cost

`cache=True` throughout, same discipline as the SA kernel; numbers in §5.
`try_attach` calls `warm_process()` unless `PARTNER_REFINE_KERNEL_WARMUP=0`.
No `inline='always'` anywhere: the kernel is four functions, three of them
already leaf-shaped, so there is nothing to buy at the 161 s compile-time
price the SA port measured for heavy inlining.
