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

# Canonical scorer for the test session: the repo copy of the official
# evaluator (scripts/iccad2026_evaluate.py). icdc_engine.qa_contract and the
# topology teacher pin its SHA256 and require `iccad2026_evaluate.__file__` to
# be that path, so import it here once -- before any test can pull the
# submodule copy under FloorSet/iccad2026contest into sys.modules first.
for path in (ROOT / "FloorSet", ROOT / "FloorSet" / "iccad2026contest", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
try:
    import iccad2026_evaluate  # noqa: F401
except Exception:  # submodule not initialised -> tests that need it skip/fail on their own
    pass
