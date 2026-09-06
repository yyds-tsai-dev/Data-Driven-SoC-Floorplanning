import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Final-submission solver modules are flat (they ship flat inside cadc1013/),
# so tests import them the same way the package does: `import layout_refiner`.
SOLVER = ROOT / "src" / "solver"
SRC = ROOT / "src"
for path in (SRC, SOLVER):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
