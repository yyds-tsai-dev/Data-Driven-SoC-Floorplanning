from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTEST_DIR = ROOT / "FloorSet" / "iccad2026contest"
for path in (SRC, CONTEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from floorset_arch.optimizer import ArchitectureV3Optimizer  # noqa: E402


MyOptimizer = ArchitectureV3Optimizer
ContestOptimizer = ArchitectureV3Optimizer
