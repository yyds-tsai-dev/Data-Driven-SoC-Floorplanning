"""`PARTNER_ORACLE_PRED_FILE` multi-layout bank (*** PROBE ONLY ***).

The oracle-injection harness used to hold ONE layout per case and broadcast
it over every replaced slot of the raw Direct batch, so a K-sample "engine"
was simulated by K identical copies.  That homogenisation is not free (it
was measured at ~+0.035 weighted no-runtime on its own), so the IC/DC Gate-0
transfer curve needs the bank form: `{"<case>": [layout, layout, ...]}` with
K DISTINCT layouts, cycled round-robin over the replaced slots.

What must hold:
  * the single-layout form is bit-identical to the pre-patch behaviour
    (broadcast, same slot count, same tail);
  * the bank form hands out distinct layouts in order and cycles when the
    bank is shorter than the batch;
  * `PARTNER_ORACLE_PRED_K` still bounds the replaced prefix in both forms;
  * every malformed / mismatched entry degrades that case to the control arm
    instead of corrupting the run;
  * with the flag unset the whole hook is a dead constant test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as csl          # noqa: E402


N = 5


def _layout(seed: float) -> list:
    return [[seed + i, seed + 2 * i, 1.0 + seed, 2.0 + seed]
            for i in range(N)]


def _batch(k: int) -> list:
    return [np.full((N, 4), -float(j), dtype=np.float64) for j in range(k)]


@pytest.fixture
def oracle(monkeypatch, tmp_path):
    """Install an oracle file and reset the module-level cache/counter."""

    def _install(mapping, case=7, k_env=None):
        path = tmp_path / "oracle.json"
        path.write_text(json.dumps(mapping))
        monkeypatch.setattr(csl, "_ORACLE_PRED_FILE", str(path))
        monkeypatch.setattr(csl, "_ORACLE_MAP", None)
        monkeypatch.setattr(csl, "_CASE_SEQ", case)
        if k_env is None:
            monkeypatch.delenv("PARTNER_ORACLE_PRED_K", raising=False)
        else:
            monkeypatch.setenv("PARTNER_ORACLE_PRED_K", str(k_env))

    return _install


# --------------------------------------------------------------- off path
def test_flag_unset_is_a_dead_constant() -> None:
    """Nothing in the shipped configuration arms the probe."""
    import os
    assert not os.environ.get("PARTNER_ORACLE_PRED_FILE")
    assert csl.ORACLE_PRED_ON is False
    assert isinstance(csl.ORACLE_PRED_ON, bool)


# ------------------------------------------------------ single-layout form
def test_single_layout_broadcasts_over_the_whole_batch(oracle) -> None:
    L = _layout(3.0)
    oracle({"7": L})
    out = csl.oracle_pred_override(_batch(4), N)
    assert len(out) == 4
    for arr in out:
        assert np.array_equal(arr, np.asarray(L, dtype=np.float64))
    # copies, not aliases
    out[0][0, 0] = 999.0
    assert out[1][0, 0] != 999.0


def test_single_layout_honours_k_prefix(oracle) -> None:
    L = _layout(3.0)
    oracle({"7": L}, k_env=2)
    batch = _batch(4)
    out = csl.oracle_pred_override(batch, N)
    assert len(out) == 4
    assert np.array_equal(out[0], np.asarray(L, dtype=np.float64))
    assert np.array_equal(out[1], np.asarray(L, dtype=np.float64))
    assert np.array_equal(out[2], batch[2])
    assert np.array_equal(out[3], batch[3])


def test_empty_batch_yields_the_single_layout(oracle) -> None:
    L = _layout(1.0)
    oracle({"7": L})
    out = csl.oracle_pred_override([], N)
    assert len(out) == 1
    assert np.array_equal(out[0], np.asarray(L, dtype=np.float64))


# --------------------------------------------------------- multi-layout bank
def test_bank_hands_out_distinct_layouts_in_order(oracle) -> None:
    bank = [_layout(float(j)) for j in range(4)]
    oracle({"7": bank})
    out = csl.oracle_pred_override(_batch(4), N)
    assert len(out) == 4
    for j, arr in enumerate(out):
        assert np.array_equal(arr, np.asarray(bank[j], dtype=np.float64))
    # the point of the patch: no two replaced slots are equal
    assert len({a.tobytes() for a in out}) == 4


def test_short_bank_cycles_round_robin(oracle) -> None:
    bank = [_layout(0.0), _layout(1.0)]
    oracle({"7": bank})
    out = csl.oracle_pred_override(_batch(5), N)
    assert len(out) == 5
    for j, arr in enumerate(out):
        assert np.array_equal(arr, np.asarray(bank[j % 2], dtype=np.float64))


def test_bank_longer_than_batch_is_truncated(oracle) -> None:
    bank = [_layout(float(j)) for j in range(6)]
    oracle({"7": bank})
    out = csl.oracle_pred_override(_batch(2), N)
    assert len(out) == 2
    assert np.array_equal(out[0], np.asarray(bank[0], dtype=np.float64))
    assert np.array_equal(out[1], np.asarray(bank[1], dtype=np.float64))


def test_bank_honours_k_prefix(oracle) -> None:
    bank = [_layout(float(j)) for j in range(4)]
    oracle({"7": bank}, k_env=2)
    batch = _batch(4)
    out = csl.oracle_pred_override(batch, N)
    assert np.array_equal(out[0], np.asarray(bank[0], dtype=np.float64))
    assert np.array_equal(out[1], np.asarray(bank[1], dtype=np.float64))
    assert np.array_equal(out[2], batch[2])
    assert np.array_equal(out[3], batch[3])


def test_bank_of_one_matches_the_single_form(oracle) -> None:
    L = _layout(2.0)
    oracle({"7": [L]})
    out = csl.oracle_pred_override(_batch(3), N)
    assert len(out) == 3
    for arr in out:
        assert np.array_equal(arr, np.asarray(L, dtype=np.float64))


def test_empty_batch_yields_the_whole_bank(oracle) -> None:
    bank = [_layout(0.0), _layout(1.0), _layout(2.0)]
    oracle({"7": bank})
    out = csl.oracle_pred_override([], N)
    assert len(out) == 3


# ------------------------------------------------------------- degradation
@pytest.mark.parametrize("bad", [
    {"9": _layout(1.0)},                       # different case id
    {"7": _layout(1.0)[:-1]},                  # wrong block count
    {"7": [[1.0, 2.0, 3.0] for _ in range(N)]},  # wrong tuple width
    {"7": []},                                 # empty
    {"7": [_layout(1.0), _layout(1.0)[:-1]]},  # ragged bank
])
def test_malformed_entries_degrade_to_control(oracle, bad) -> None:
    oracle(bad)
    batch = _batch(3)
    out = csl.oracle_pred_override(batch, N)
    assert out is batch


def test_unreadable_file_degrades_to_control(monkeypatch, tmp_path) -> None:
    path = tmp_path / "missing.json"
    monkeypatch.setattr(csl, "_ORACLE_PRED_FILE", str(path))
    monkeypatch.setattr(csl, "_ORACLE_MAP", None)
    monkeypatch.setattr(csl, "_CASE_SEQ", 7)
    batch = _batch(3)
    assert csl.oracle_pred_override(batch, N) is batch
    assert csl._ORACLE_MAP == {}
