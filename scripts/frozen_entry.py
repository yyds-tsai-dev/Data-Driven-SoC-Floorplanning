"""Frozen (PyInstaller) contest entry point for the column-backbone solver.

Contract (mirrors scripts/op_wrapper.py):
  stdin  -> one JSON case payload:
            {"block_count": int, "area_targets": [...], "b2b_connectivity": [...],
             "p2b_connectivity": [...], "pins_pos": [...], "constraints": [...],
             "target_positions": [...] | null}
  stdout -> {"positions": [[x, y, w, h], ...]}   (nothing else on stdout)
  stderr -> free-form diagnostics (timings, pool status)

Modes:
  (default)              read payload from stdin, solve, print JSON
  --payload FILE         read payload from FILE instead of stdin
  --startup-probe        import solver + warm worker pool, print startup JSON, exit

Notes for the frozen build (see docs/design/packaging_notes.md):
  * multiprocessing.freeze_support() MUST run first under __main__: with the
    'spawn' start method a PyInstaller child re-executes this binary, and
    freeze_support() is what diverts that re-execution into worker mode
    instead of a recursive solve. The production pool uses 'fork'
    (src/floorset_arch/legalizer/column_slicing.py::init_worker_pool), for
    which this is a no-op on Linux, but the guard is required for safety.
  * The production path is pure CPU. FLOORSET_COLUMN_BACKBONE is forced to 1
    and FLOORSET_DIFFUSION_CHECKPOINT is dropped so the frozen binary can
    never wander into a model-checkpoint path (no checkpoints are bundled).
  * A snapshot of the repo .env is bundled next to the executable and applied
    with setdefault semantics (caller environment wins), matching
    optimizer._load_repo_dotenv(override=False).
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[frozen_entry] {msg}", file=sys.stderr, flush=True)


def _apply_env_defaults() -> None:
    """Load the bundled .env snapshot (setdefault semantics), then pin the
    frozen binary to the pure-CPU production path."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    else:
        # Unfrozen (parity/debug) runs use the repo .env and repo src/.
        base = Path(__file__).resolve().parents[1]
        src = base / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    env_file = base / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
    # Hard floor for the frozen artifact: always the CPU column backbone,
    # never a checkpoint-consuming path, never CUDA.
    os.environ["FLOORSET_COLUMN_BACKBONE"] = "1"
    os.environ.pop("FLOORSET_DIFFUSION_CHECKPOINT", None)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _tensor(torch, data, empty_cols: int):
    if data is None:
        return None
    if isinstance(data, list) and len(data) == 0:
        return torch.empty((0, empty_cols), dtype=torch.float32)
    return torch.tensor(data, dtype=torch.float32)


def main(argv: list[str]) -> int:
    # Keep solver chatter off stdout: the op_wrapper contract allows exactly
    # one JSON object there. Restore stdout only for the final dump.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    t0 = time.time()
    _apply_env_defaults()

    import torch  # noqa: F401  (heaviest import; timed separately)

    t_torch = time.time()

    from floorset_arch.optimizer import ArchitectureV11Optimizer

    t_solver = time.time()

    # __init__ warms the fork worker pool (warm_worker_pool) so SA restarts
    # do not pay spawn cost inside the solve budget.
    optimizer = ArchitectureV11Optimizer(verbose=False)
    t_init = time.time()

    from floorset_arch.legalizer import column_slicing as _cs

    pool_size = int(getattr(_cs, "_POOL_SIZE", 0))
    pool_ready = bool(getattr(_cs, "_POOL_READY", False))
    _log(
        f"frozen={bool(getattr(sys, 'frozen', False))} "
        f"import_torch_s={t_torch - t0:.3f} import_solver_s={t_solver - t_torch:.3f} "
        f"init_pool_s={t_init - t_solver:.3f} startup_total_s={t_init - t0:.3f} "
        f"pool_size={pool_size} pool_ready={pool_ready}"
    )

    if "--startup-probe" in argv:
        sys.stdout = real_stdout
        json.dump(
            {
                "frozen": bool(getattr(sys, "frozen", False)),
                "startup_s": round(t_init - t0, 3),
                "import_torch_s": round(t_torch - t0, 3),
                "import_solver_s": round(t_solver - t_torch, 3),
                "init_pool_s": round(t_init - t_solver, 3),
                "pool_size": pool_size,
                "pool_ready": pool_ready,
            },
            real_stdout,
        )
        real_stdout.write("\n")
        return 0

    if "--payload" in argv:
        payload_path = argv[argv.index("--payload") + 1]
        with open(payload_path, "r") as fh:
            payload = json.load(fh)
    else:
        payload = json.load(sys.stdin)

    block_count = int(payload["block_count"])
    t_parse = time.time()
    positions = optimizer.solve(
        block_count,
        _tensor(torch, payload["area_targets"], 1),
        _tensor(torch, payload["b2b_connectivity"], 3),
        _tensor(torch, payload["p2b_connectivity"], 3),
        _tensor(torch, payload["pins_pos"], 2),
        _tensor(torch, payload["constraints"], 5),
        _tensor(torch, payload.get("target_positions"), 4),
    )
    t_solve = time.time()
    _log(f"n={block_count} solve_s={t_solve - t_parse:.3f} total_s={t_solve - t0:.3f}")

    sys.stdout = real_stdout
    json.dump(
        {"positions": [[float(v) for v in rect] for rect in positions]},
        real_stdout,
    )
    real_stdout.write("\n")
    return 0


if __name__ == "__main__":
    # MUST be first under __main__ in a frozen binary (spawn-safety; see
    # module docstring). No-op for the production fork pool on Linux.
    multiprocessing.freeze_support()
    sys.exit(main(sys.argv[1:]))
