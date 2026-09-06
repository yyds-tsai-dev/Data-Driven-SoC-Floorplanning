# `PARTNER_FAST_SETUP` -- budget-aware per-case fixed cost

Status: implemented, **default off**, no evaluator evidence yet (single-thread
anatomy + single-case pool micro-benchmark + bit-exactness evidence only; the
paired full-100 A/B is the scheduler's).
Flag surface: `PARTNER_FAST_SETUP=1`.
Code: `src/solver/column_sa_legalizer.py` (`fast_setup_on`, the anneal-span floor
in `_ColumnOptimizer.finish`, `_pin_centroids_np`, `_b2b_smooth_np`),
`src/solver/contest_optimizer.py` (`_heuristic_init`).
Tests: `tests/test_partner_fast_setup.py` (18).
Instruments: `scratchpad/floor_anatomy.py`,
`scratchpad/floor_finish_profile.py`, `scratchpad/floor_seed_probe.py`,
`scratchpad/floor_seed_equiv.py`, `scratchpad/floor_wall_bench.py`,
`scratchpad/floor_quality_probe.py`, `scratchpad/floor_projection.py`.
Prior art: `docs/experiments/2026-08-04-low-budget-frontier-pool-gate.md`
(`PARTNER_POOL_GATE`, the cliff this sits under),
`docs/design/2026-08-04-early-exit-true-time-reduction.md` (time anatomy
method).

---

## 0. The question

At the goal operating point -- `b(n) = 5e-5 * exp(n/12)` clamped to
[0.05, 0.75], `PARTNER_POOL_GATE=0`, both numba kernels on -- a small case
**asks for 0.05 s and measures 0.12-0.17 s of wall clock**, independent of
its budget. The suspects on entry were (1) the 24-way payload marshal, (2)
per-worker `_ColumnOptimizer` build, (3) deadline granularity.

Suspects 1 and 2 are wrong. Suspect 3 is the whole story, and it is not
granularity -- it is a **hard-coded 0.1 s minimum anneal span**.

## 1. Floor anatomy

Method: single process, single thread, no pool, no evaluator. Replay exactly
what `legalize_rectangles -> _parallel_solve -> _worker_solve` do for one case
at `budget = 0.05`, timing each stage separately (`floor_anatomy.py`, median
of 7 reps; `floor_finish_profile.py` for the stage split inside `finish`).

| stage | n=25 | n=50 | n=70 | where |
|---|---|---|---|---|
| heuristic seed (`_heuristic_init`) | 3.8 ms | 4.3 ms | 10.7 ms | parent, serial |
| parent `_ColumnOptimizer` build | 0.75 ms | 1.16 ms | 1.61 ms | parent, serial |
| -- of which numba kernel attach | 0.15 ms | 0.13 ms | 0.16 ms | parent, serial |
| payload marshal (24 configs) | 0.24 ms | 0.24 ms | 0.24 ms | parent, serial |
| pickle of all 24 payloads | 0.35 ms | 0.42 ms | 0.46 ms | dispatch |
| unpickle (1 payload) | 0.01 ms | 0.01 ms | 0.02 ms | worker |
| worker tensor cast | 0.01 ms | 0.01 ms | 0.01 ms | worker |
| worker `_ColumnOptimizer` build | 0.73 ms | 1.20 ms | 1.68 ms | worker |
| worker `prepare()` | 0.10 ms | 0.13 ms | 0.20 ms | worker |
| **worker `finish()`** | **101.1 ms** | **102.0 ms** | **104.8 ms** | worker |
| -- planned span for that `finish` | 41.4 ms | 41.0 ms | 40.5 ms | |
| -- **overshoot** | **+59.7 ms** | **+61.1 ms** | **+64.3 ms** | |
| payload size (bytes) | 5647 | 8742 | 17510 | |

Verdicts:

* **The 24x payload replication is a non-issue.** Marshal is 0.24 ms and the
  whole 24-payload pickle is 0.35-0.46 ms, because the arrays are small
  (6-18 KB per payload) and `np_of` returns views. Shared memory / a pool
  initializer broadcast would buy < 0.5 ms/case. Not worth the machinery.
* **Per-worker build is a non-issue.** Build + prepare is 0.8-1.9 ms, and it
  is parallel across workers. Vectorising it would buy < 2 ms/case.
* **`finish` overshoots its deadline by 2.5x**, and `floor_finish_profile.py`
  puts 97.8-99.1 % of that inside a single `_anneal` call that runs
  100.6-100.9 ms against a ~30 ms plan.

## 2. Root cause

`finish` (`src/solver/column_sa_legalizer.py`):

```python
refine_t = min(0.30 * rem, 7.0) if rem > 1.0 else 0.0
polish_t = min(0.28 * max(rem - refine_t, 0.0), 6.5)
t_end    = deadline - polish_t - refine_t
now      = time.time()
total    = max(t_end - now, 0.1)      # <-- flat 0.1 s floor
```

The floor is longer than the entire case budget at the goal tier, so every
one of the ~24 pool workers anneals ~100 ms no matter what deadline it was
handed. Wall clock = 100 ms + ~15 ms of everything else = the observed
0.12-0.17 s. The budget curve's `PARTNER_BUDGET_MIN` knob is inoperative
below ~0.165 s: setting it lower changes nothing but the label.

Second finding, from the same anatomy: `_heuristic_init`'s two connectivity
loops walk the **padded** `b2b` / `p2b` tensors row by row with three
`.item()` calls each. Cost tracks tensor height, not instance size
(`floor_seed_probe.py`):

| n | 25 | 50 | 70 | 100 | 120 |
|---|---|---|---|---|---|
| seed | 4.7 ms | 4.9 ms | 12.1 ms | 11.1 ms | **138.0 ms** |

All of it serial in the parent, before any worker sees the case -- 18 % of
the budget on the `n=120` tail case.

## 3. What the flag does

Two surgeries, one flag, both no-ops when off.

**3.1 Anneal-span floor (the 0.12-0.17 s floor).**

```python
_span_floor = 0.1
if fast_setup_on():
    _span_floor = min(_span_floor, max(t_end - now, 0.0))
total = max(t_end - now, _span_floor)
```

The clamp can only bind when the planned span is already below 0.1 s. Above
that it is arithmetically inert, so **a case at any pre-goal-tier budget
keeps its exact anneal span with the flag ON** (asserted in
`test_flag_is_a_no_op_above_the_floor`). This is not overhead removal: it is
removing search the case never asked for, and it makes `PARTNER_BUDGET_MIN`
an operative knob again.

**3.2 Bit-exact numpy seed (`_pin_centroids_np`, `_b2b_smooth_np`).**

Transcriptions of the two edge loops. The contract is **bit equality**, not
tolerance -- the seed feeds every restart, so anything less would turn a
speed change into a search-trajectory change. Two properties carry it:

* `np.add.at` is *unbuffered*, so repeated target indices accumulate in index
  order, which is the loop's edge order;
* the b2b loop writes `nx[i]` before `nx[j]` per edge, so the flattened index
  stream must be **interleaved** (`i0, j0, i1, j1, ...`), not concatenated --
  concatenation reverses the relative order for any node that appears as `i`
  in a later edge and as `j` in an earlier one.

Both helpers return `None` on any unexpected tensor layout, and the caller
keeps the reference loops.

## 4. Micro-benchmark

Per-case wall clock, one arm per process (the pool is forked *after* the arm
env is set, as the submission wrapper does), real 24-worker restart pool,
5 reps, median, no evaluator (`floor_wall_bench.py`):

| tid | n | budget | seed off->on | `legalize_rectangles` off->on | case off->on | saved |
|---|---|---|---|---|---|---|
| 4 | 25 | 0.050 | 3.7 -> 1.4 ms | 107.6 -> 44.9 ms | **111.6 -> 46.3 ms** | -58 % |
| 29 | 50 | 0.050 | 4.4 -> 1.1 ms | 111.1 -> 45.7 ms | **115.4 -> 46.8 ms** | -59 % |
| 49 | 70 | 0.050 | 11.0 -> 1.6 ms | 115.1 -> 47.3 ms | **125.9 -> 48.9 ms** | -61 % |
| 79 | 100 | 0.208 | 10.1 -> 2.1 ms | 184.0 -> 184.7 ms | **194.1 -> 186.8 ms** | -4 % |
| 99 | 120 | 0.750 | 123.9 -> 2.9 ms | 648.2 -> 648.9 ms | **772.8 -> 651.8 ms** | -16 % |

The floor-budget cases now land within 3 ms of the budget they asked for
(46-49 ms against 50 ms). The two tail cases show the clamp is inert there --
their whole gain is the seed.

Seed bit-equality and cost over the real validation set
(`floor_seed_equiv.py`): **100/100 cases bit-identical**, 2.41 s -> 0.16 s.

Projected full-run saving (`floor_projection.py`): the anneal floor binds on
**77/100** cases; 14.31 s -> 9.39 s of anneal plus 2.25 s of seed =
**~7.2 s per goal-tier run** (which totals ~21 s today).

## 5. What the reclaimed time costs

The clamp shortens every affected chain from ~100 ms to ~31 ms. A
single-thread proxy for the pool's min-over-restarts -- K=12 restarts per
arm, scored with `_parallel_solve`'s own true-cost rule on a common
normalizer (`floor_quality_probe.py`) -- finds essentially nothing to lose:

| tid | n | best score 100 ms -> 31 ms | V | hpwl |
|---|---|---|---|---|
| 4 | 25 | 4.10591 -> 4.10591 (+0.00 %) | 7 -> 7 | 60 -> 60 |
| 19 | 40 | 1.13390 -> 1.13262 (-0.11 %) | 1 -> 1 | 23 -> 23 |
| 29 | 50 | 1.47341 -> 1.47196 (-0.10 %) | 5 -> 5 | 30 -> 30 |
| 39 | 60 | 1.11383 -> 1.11383 (+0.00 %) | 2 -> 2 | 108 -> 108 |
| 49 | 70 | 1.19325 -> 1.19325 (+0.00 %) | 3 -> 3 | 99 -> 99 |
| 64 | 85 | 1.25696 -> 1.25825 (+0.10 %) | 5 -> 5 | 209 -> 204 |

Read: at these instance sizes the chains have already converged well before
30 ms, so the extra 70 ms was buying nothing measurable. **Caveat:** the
probe varies only the seed, while the real portfolio also varies column
count / orientation / `v_weight` / `h_scale`, and 12 is not 24 restarts. It
is a proxy, not evaluator evidence.

## 6. Off-path contract

* `fast_setup_on()` reads `PARTNER_FAST_SETUP` (default `0`) through the
  shared `_flag_on` spelling set; nothing else in either module changed.
* Off, the seed takes the original loops (`_got is None` -> the `for edge in
  ...` bodies run verbatim) and `total = max(t_end - now, 0.1)` is the
  original expression.
* Full suite with the flag unset: 851 passed, 1 pre-existing failure
  (`test_optimizer.py::test_checkpoint_relative_path_resolves_from_repo_root`,
  which also fails on the clean base in this worktree).

## 7. What is left on the floor (measured, not done)

* Payload sharing / pool-initializer broadcast: < 0.5 ms/case. Not worth it.
* Worker build + prepare vectorisation: < 2 ms/case, and parallel. Not worth
  it.
* Parent-side `_ColumnOptimizer` kernel attach when the pool path will be
  taken (`opt1` never anneals): 0.13-0.16 ms/case. Not worth it.
* `PARTNER_SA_KERNEL` is not warmed before the fork the way
  `PARTNER_REFINE_KERNEL` is (`init_worker_pool`), so every worker pays its
  own numba cache load once, on the first case it handles. One-time, not part
  of the steady floor -- but it is a free `warm_process()`-style fix if the
  first-case tail ever matters.

## 8. How to decide it

The flag is two claims with different evidence bars:

1. the seed twin is bit-exact (proved: 100/100 real cases + unit tests) and
   pure profit -- it should be safe at every tier;
2. the anneal clamp trades ~70 ms/case of converged search for wall clock on
   77/100 cases -- the proxy says the trade is free, but only a **paired
   full-100 at the goal tier** (`total_score_no_runtime` + raw runtime tail)
   can promote it. The natural follow-up A/B is the one that *spends* the
   reclaimed ~7 s: re-run the goal tier with `PARTNER_FAST_SETUP=1` at a
   raised `PARTNER_BUDGET_SCALE` / tail budget so the average per-case wall
   returns to today's ~0.21 s.
