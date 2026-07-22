import importlib.util
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_builder_module():
    path = REPO_ROOT / "scripts" / "build_partner_retrieval_index.py"
    spec = importlib.util.spec_from_file_location("build_partner_retrieval_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retrieval_builder_help_is_noninteractive_and_exposes_required_flags():
    result = subprocess.run(
        [sys.executable, "scripts/build_partner_retrieval_index.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for flag in ("--data-path", "--output", "--max-per-n", "--seed"):
        assert flag in result.stdout


def test_probe_help_is_noninteractive_and_exposes_required_flags():
    result = subprocess.run(
        [sys.executable, "scripts/probes/retrieval_probe.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for flag in ("--index", "--data-path", "--cases", "--top-k", "--output"):
        assert flag in result.stdout


def test_builder_is_hard_coded_to_the_training_split_only():
    text = (REPO_ROOT / "scripts" / "build_partner_retrieval_index.py").read_text()

    assert "--source-split" not in text
    assert 'source_split="train"' in text
    assert "FloorplanDatasetLiteTest" not in text


def test_reservoir_sampling_is_deterministic_and_capped_per_block_count():
    builder = _load_builder_module()
    block_counts = [2] * 20 + [3] * 10 + [4]

    first = builder.reservoir_indices_by_block_count(block_counts, max_per_n=3, seed=17)
    second = builder.reservoir_indices_by_block_count(block_counts, max_per_n=3, seed=17)

    assert first == second
    assert {n: len(indices) for n, indices in first.items()} == {2: 3, 3: 3, 4: 1}
    assert all(0 <= index < 20 for index in first[2])
    assert all(20 <= index < 30 for index in first[3])
    assert first[4] == [30]
