# PyInstaller packaging notes (frozen submission harness)

2026-07-08. Deliverables: `scripts/package_submission.sh` (build),
`scripts/frozen_entry.py` (frozen entry point), `scripts/smoke_frozen.sh`
(frozen-vs-native smoke). Output lands in `scripts/dist/` (gitignored via the
`dist/` pattern); the packaging venv lives in `build/pack-venv` (gitignored).
Nothing under `src/` was changed; `pyproject.toml` was NOT modified —
pyinstaller is installed only into the packaging venv.

## Contract

`scripts/op_wrapper.py` (contest-side wrapper) spawns the binary **per case**:
JSON payload on stdin -> `{"positions": [[x,y,w,h], ...]}` on stdout, 60 s
`subprocess.run` timeout, binary expected at
`scripts/dist/my_optimizer/my_optimizer` (onedir). `frozen_entry.py` follows
that contract exactly and keeps stdout clean by redirecting all solver chatter
to stderr until the final JSON dump.

## Pitfalls found and how they are handled

1. **torch cannot be excluded.** The production import chain is *not*
   torch-free: `optimizer.py` imports torch at module top, and the column
   backbone uses it at runtime for tensor plumbing (`torch.full` in
   `column_backbone.py`, `torch.from_numpy` in every SA worker in
   `column_slicing.py`, `torch.as_tensor` in `parser.py`). Excluding torch
   would need a numpy-backed shim over ~10 call sites — rejected for
   correctness risk.
2. **The main `.venv` torch is unpackageable.** It is `2.6.0+cu124`;
   `libtorch_python.so` hard-links `libtorch_cuda.so`, which resolves into
   ~4.7 GB of `nvidia/*` pip libs (verified with `ldd`). PyInstaller would
   bundle all of it. Fix: dedicated packaging venv (`build/pack-venv`) with
   **`torch==2.6.0+cpu`** from `https://download.pytorch.org/whl/cpu`, plus
   `numpy==1.26.4` / `shapely==2.0.5` pinned to the main-venv versions so the
   frozen solver computes on identical numerics. `package_submission.sh`
   aborts if the packaging venv's torch is not a `+cpu` build.
3. **`freeze_support()`.** First statement under `__main__` in
   `frozen_entry.py`. Required for spawn-method safety in any frozen binary
   (a spawn child re-executes the executable). The production pool uses
   **fork** (`column_slicing.init_worker_pool`), which works unchanged when
   frozen — measured: `pool_size=32 pool_ready=True` inside the frozen
   binary, and 2326 % CPU on the n=103 case (see smoke below).
4. **Contest harness not needed in the bundle.** `optimizer.py`'s
   `from iccad2026_evaluate import FloorplanOptimizer` is try/except-guarded
   with a local stub, so the frozen bundle ships without
   `FloorSet/iccad2026contest`.
5. **Config self-containment.** The repo `.env` is bundled (`--add-data`) and
   applied by the entry with setdefault semantics (caller env wins, same as
   `load_dotenv(override=False)`). The entry then **forces**
   `FLOORSET_COLUMN_BACKBONE=1`, drops `FLOORSET_DIFFUSION_CHECKPOINT`, and
   sets `CUDA_VISIBLE_DEVICES=""` — the frozen artifact can never wander into
   a model-checkpoint path (no checkpoints are bundled).
6. **Hidden imports.** PyInstaller's bytecode scan does find the
   function-level imports (`legalizer.column_backbone`, `refine.api`, ...),
   but they are also listed explicitly in `package_submission.sh` so a future
   refactor to dynamic imports cannot silently drop the legalizer.
   The `torch.utils.tensorboard` collect warning during build is benign.

## Measured startup (per-case spawn cost)

48-core host, warm FS cache, 3 runs each (`--startup-probe` = imports +
optimizer init + fork-pool warmup, timed externally):

| variant                | bundle size | wall/spawn  | breakdown (internal) |
|------------------------|------------|--------------|----------------------|
| onedir (submission)    | 732 MB     | **2.09–2.13 s** | bootloader ~0.4 s + `import torch` ~1.0 s + solver import 0.02 s + pool warm 0.66 s |
| onefile (comparison)   | 215 MB     | 5.01–5.07 s  | +~3.4 s self-extraction per spawn, identical internals |

- **onedir wins** — the onefile extraction penalty is paid on *every* case
  because op_wrapper spawns per case. Onefile exists only for comparison
  (`--onefile` flag).
- Cold-cache caveat: the very first onedir run on NFS took 21.5 s
  (`import torch` paging 732 MB of libs over NFS). One-off; irrelevant on the
  contest machine's local disk, but pre-warm with one `--startup-probe` run
  before timed evaluations.

## Smoke results (frozen vs native, official evaluator)

`bash scripts/smoke_frozen.sh`, cases 0 / 37 / 82 (n = 21 / 58 / 103), frozen
driven through the real `op_wrapper` spawn contract, native = in-process
`src/architecture_v11_optimizer.py`, same `.env` toggles both sides:

| case | n   | feasible F/N | cost frozen | cost native | delta | runtime F | runtime N |
|------|-----|--------------|-------------|-------------|-------|-----------|-----------|
| 0    | 21  | T / T        | 1.3824      | 1.3757      | +0.5 % | 6.87 s    | 0.42 s    |
| 37   | 58  | T / T        | 1.2212      | 1.1927      | +2.4 % | 3.07 s    | 1.00 s    |
| 82   | 103 | T / T        | 1.4894      | 1.4911      | −0.1 % | 22.53 s   | 18.26 s   |

- Costs are in-band (SA restart noise is larger than these deltas);
  `cost_no_runtime` matched `cost` in all rows.
- Parallelism proof (direct spawn under `/usr/bin/time -v`): n=103 solve ran
  at **2326 % CPU** (fork pool active in frozen mode); small cases sit at
  ~270 % because their SA budget (~0.8–1.1 s) ends before the portfolio fans
  out.
- Case 0's frozen runtime (6.87 s vs ~2.6 s expected) was a first-spawn NFS
  page-cache artifact inside the evaluator run; the direct spawn of the same
  payload took 2.52 s wall.

## Submission-decision implications

- Per-case spawn overhead is **~2.1–2.7 s** with onedir. Over 100 hidden
  cases that is ~3.5–4.5 min of added *per-case* runtime. RuntimeFactor is
  per-case vs field median (see CONTEXT.md): on small cases the overhead
  multiplies measured runtime by >5x (0.4 s -> ~2.6+ s), so if the contest
  permits an in-process Python submission, that path avoids the tax entirely;
  the frozen binary is the fallback that removes all environment risk.
- **op_wrapper timeout raised 60 s -> 300 s (2026-07-08):** the 60 s was OUR
  wrapper's own choice, not a contest rule (verified 2026-07-07: no official
  per-case limit; slow = uncapped RTF penalty, timeout = raise -> infeasible
  -> cost 10, strictly worse). With the promoted E2 tail budget the worst case
  was ~51 s incl. spawn — a ~9 s self-inflicted landmine margin. 300 s is a
  pure hang guard; any adaptive-budget cap must still stay far below it.
- Startup is torch-import-bound (~1.0 s of the 2.1 s). A numpy-only shim for
  the four torch call sites in the backbone chain is the obvious next lever
  if per-case spawn cost must shrink further (it also unlocks a smaller
  typed-C-ext-friendly bundle).

## How to rebuild / re-verify

```bash
bash scripts/package_submission.sh            # onedir only
bash scripts/package_submission.sh --onefile  # + onefile comparison variant
bash scripts/smoke_frozen.sh                  # gates: all frozen cases feasible
CASES="0 99" bash scripts/smoke_frozen.sh     # custom cases (99 = n=120 tail)
```

Smoke artifacts (payloads, per-case evaluator JSONs, `/usr/bin/time` logs)
land in `artifacts/frozen_smoke/`.
