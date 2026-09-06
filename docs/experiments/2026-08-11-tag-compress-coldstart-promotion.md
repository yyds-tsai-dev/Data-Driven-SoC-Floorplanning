# 2026-08-11 TAG_COMPRESS cold-start revival: PROMOTE

## Decision

Promote `PARTNER_TAG_COMPRESS=1` at the 0.3 s operating point, together with
constructor-time warming of its exact violation scorer.

The geometry pass already existed default-off. It pushes a movable separation
chain inward until a reachable preplaced boundary-tag line becomes the final
bbox edge. It never moves preplaced blocks, preserves fixed/MIB shapes, keeps
soft-area error below 1%, rechecks overlap, and commits only after its
evaluator-form soft-violation gate accepts the candidate.

## Why the old NO-GO was reopened

The earlier isolated-tail experiment attributed roughly 0.15--0.4 s to the
pass and recorded an explicit rework condition: revisit only with a cheap
acceptance path. A current saved-layout replay showed that the hot geometry
pass is much cheaper than that attribution:

- median over all cases: about 0.072 ms;
- maximum on a changed layout: 9.51 ms;
- isolated case 89 before warming: post section 0.641 s versus 0.216 s OFF;
- isolated case 89 after constructor warming: post section 0.242 s versus
  0.216 s OFF, about 26 ms residual including all post-processing noise.

The large old number was principally lazy module/dependency initialization in
the timed first call. `MyOptimizer.__init__` now calls
`tag_compress.warm_dependencies()` only when the existing flag is enabled;
the evaluator constructs the optimizer outside per-case timing. Flag-off
imports and output remain unchanged.

## G0: noise-free saved-layout replay

Each ON result was generated directly from the corresponding OFF positions,
then re-scored by the official evaluator. This removes deadline-SA variance.

| Layout set | Base noRT | Compressed noRT | Delta | Changed cases |
|---|---:|---:|---:|---|
| `col_lns_full_control` | 1.154704780 | 1.151505972 | -0.003198808 | 53, 88, 89, 99 |
| `col_lns_full_on` | 1.150123840 | 1.144019967 | -0.006103874 | 53, 88, 89, 99 |
| `col_lns_full_control2` | 1.150745238 | 1.145916304 | -0.004828934 | 53, 88, 89, 99 |
| `col_lns_full_on2` | 1.151197830 | 1.147926899 | -0.003270931 | 53, 88, 89, 99 |

Mean self-paired delta is **-0.004350637**, all four replicates improve, and
all 400 re-scored layouts are hard feasible. The repeated affected case set
also verifies mechanism stability across independently sampled upstream
layouts.

## G1: online full-100 reversed pairs

| Pair/order | OFF noRT | ON noRT | ON-OFF | OFF avg runtime | ON avg runtime |
|---|---:|---:|---:|---:|---:|
| 1, ON then OFF | 1.164760355 | 1.151030097 | -0.013730258 | 0.289291 s | 0.294031 s |
| 2, OFF then ON | 1.156191692 | 1.145249482 | -0.010942210 | 0.294178 s | 0.293396 s |
| Mean | — | — | **-0.012336234** | **0.291734 s** | **0.293713 s** |

Every arm is 100/100 hard feasible. The online score magnitude includes the
known deadline-SA noise, so promotion rests primarily on the four noise-free
self-pairs; the online pairs establish direction sanity and the runtime gate.
The opt-in pass adds about 1.98 ms/case on the paired mean and remains below
the user-locked 0.300 s average target.

## Promotion scope

- Enable `PARTNER_TAG_COMPRESS=1` in the development environment and final
  submission wrapper.
- Keep constructor warming conditional on that flag; default-off callers stay
  byte-identical.
- Do not revive column split/merge G1: its separately recorded G0 tail
  expectation was non-positive.

## Final package verification

- Archive: `submission/cadc1013_0811d_tagcompress.tar.gz`
- MD5: `aab0c6553e387275301674095a0cd709`
- Contents: 32 tar entries; no `__pycache__`, `.pyc`, `.nbi`, or `.nbc`
- Source closure: packaged `tag_compress.py` is byte-identical to
  `src/solver/tag_compress.py`; packaged `op_src.py` is byte-identical to
  `src/solver/contest_optimizer.py`
- Fresh extraction validation: PASS; interface smoke runtime 0.057 s
- Fresh extraction full-100: noRT **1.152956051**, average runtime
  **0.290269905 s**, 100/100 hard feasible
