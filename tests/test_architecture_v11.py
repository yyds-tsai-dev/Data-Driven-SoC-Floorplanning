import importlib.util
from pathlib import Path

from floorset_arch.optimizer import (
    ArchitectureV4Optimizer,
    ArchitectureV5Optimizer,
    ArchitectureV11Optimizer,
)


def _load_wrapper(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v11_optimizer_aliases_preserve_legacy_names():
    assert ArchitectureV5Optimizer is ArchitectureV11Optimizer
    assert ArchitectureV4Optimizer is ArchitectureV11Optimizer


def test_v11_wrapper_exports_contest_optimizer():
    module = _load_wrapper(Path("src/architecture_v11_optimizer.py"))
    assert module.MyOptimizer is ArchitectureV11Optimizer
    assert module.ContestOptimizer is ArchitectureV11Optimizer


def test_v5_wrapper_forwards_to_v11_optimizer():
    module = _load_wrapper(Path("src/architecture_v5_optimizer.py"))
    assert module.MyOptimizer is ArchitectureV11Optimizer
    assert module.ContestOptimizer is ArchitectureV11Optimizer


def test_active_scripts_reference_v11_wrapper():
    for script in ["scripts/validate.sh", "scripts/eval_single.sh", "scripts/eval_total.sh"]:
        text = Path(script).read_text(encoding="utf-8")
        assert "architecture_v11_optimizer.py" in text
