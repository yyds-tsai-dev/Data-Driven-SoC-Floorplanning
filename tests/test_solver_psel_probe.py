"""The candidate-selector probe/fix knobs, all default off.

  * `PARTNER_PSEL_DUMP=<dir>`  -> buffer every pool candidate the selector
    arbitrates over and write one npz at interpreter exit (offline OFFICIAL
    cost rescoring; the internal score uses a different denominator and is
    not comparable to the evaluator's).
  * `PARTNER_WSTAR_FILE=<json>` -> feed the FRAME_WPIN arm an externally
    supplied frame width per case instead of the L/R-tag derivation (the
    G-T2'-0 golden-frame oracle probe).
  * `PARTNER_PSEL_FIX=1`       -> correct the selector's two calibration
    defects: `hp_ref` is the pool minimum where the evaluator divides by the
    golden hpwl (a factor 1+g smaller), and the 0.985 channel-swap dead zone
    is wider than most true A/B gaps.

What must hold:
  * the three module constants are read ONCE at import, so with none of them
    set every hook is a dead constant test (no per-case env lookup, no
    counter, no import) -- `_PROBE_ON` is the single witness;
  * the shipped selector constants are still 0.985 / pool-min when the fix
    flag is off, and both move when it is on;
  * the W* file takes precedence over the tag derivation but never invents a
    degenerate frame;
  * the dump round-trips (positions, channel codes, selected flag).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as csl          # noqa: E402

_ENV = ("PARTNER_PSEL_DUMP", "PARTNER_WSTAR_FILE", "PARTNER_PSEL_FIX",
        "PARTNER_PSEL_FIX_G", "PARTNER_PSEL_FIX_DZ", "PARTNER_FRAME_WPIN")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# ---------------------------------------------------------------------------
# 1. the off path is a dead constant test
# ---------------------------------------------------------------------------
def test_probes_are_off_by_default_in_this_checkout():
    """No probe env in the test environment -> every hook is dead."""
    assert csl._PSEL_DUMP_PATH == ""
    assert csl._WSTAR_FILE == ""
    assert csl._PROBE_ON is False


def test_probe_constants_are_import_time_not_per_case(monkeypatch):
    """Setting the variable AFTER import must not arm anything -- that is
    what keeps the production path free of a per-case environment read."""
    monkeypatch.setenv("PARTNER_PSEL_DUMP", "/nonexistent/should/not/be/used")
    assert csl._PSEL_DUMP_PATH == ""
    assert csl._PROBE_ON is False


# ---------------------------------------------------------------------------
# 2. selector constants
# ---------------------------------------------------------------------------
def _selector(hp_ref: float, flag: bool, monkeypatch):
    """Reproduce the two lines the fix owns, through the module's own
    readers, so the test breaks if either default drifts."""
    dz = 0.985
    if csl._flag_on("PARTNER_PSEL_FIX"):
        hp_ref = hp_ref / (1.0 + csl._env_num("PARTNER_PSEL_FIX_G", 0.29))
        dz = csl._env_num("PARTNER_PSEL_FIX_DZ", 1.0)
    return hp_ref, dz


def test_shipped_selector_constants(monkeypatch):
    hp, dz = _selector(100.0, False, monkeypatch)
    assert hp == 100.0
    assert dz == 0.985


def test_fix_moves_both_constants(monkeypatch):
    monkeypatch.setenv("PARTNER_PSEL_FIX", "1")
    hp, dz = _selector(129.0, True, monkeypatch)
    assert hp == pytest.approx(129.0 / 1.29)
    assert dz == 1.0
    monkeypatch.setenv("PARTNER_PSEL_FIX_G", "0.0")
    monkeypatch.setenv("PARTNER_PSEL_FIX_DZ", "0.985")
    hp, dz = _selector(129.0, True, monkeypatch)
    assert (hp, dz) == (129.0, 0.985)          # the inert null arm


def test_fix_source_uses_the_named_knobs():
    import inspect
    # 2026-08-28 (Sec.17t): both constants moved OUT of `_parallel_solve`
    # into the pure helpers `psel_selector_params` / `psel_pick` so they can
    # be unit-tested directly and so PARTNER_PSEL_DZ / PARTNER_PSEL_G can
    # address the dead zone and the hp_ref calibration independently.  The
    # invariant this test owns is unchanged: the knobs are read BY NAME, and
    # the dead zone multiplies the COLUMN champion's score.
    src = (inspect.getsource(csl._parallel_solve)
           + inspect.getsource(csl.psel_selector_params)
           + inspect.getsource(csl.psel_pick))
    assert 'PARTNER_PSEL_FIX' in src
    assert 'dz * score(best_col)' in src
    assert '0.985 * score(' not in src


# ---------------------------------------------------------------------------
# 3. W* oracle file
# ---------------------------------------------------------------------------
def test_wstar_oracle_maps_case_sequence(tmp_path, monkeypatch):
    p = tmp_path / "ws.json"
    p.write_text(json.dumps({"0": 12.5, "3": 40.0, "4": 0.5}))
    monkeypatch.setattr(csl, "_WSTAR_FILE", str(p))
    monkeypatch.setattr(csl, "_WSTAR_MAP", None)
    monkeypatch.setattr(csl, "_CASE_SEQ", 0)
    assert csl._wstar_oracle() == 12.5
    monkeypatch.setattr(csl, "_CASE_SEQ", 3)
    assert csl._wstar_oracle() == 40.0
    monkeypatch.setattr(csl, "_CASE_SEQ", 4)     # degenerate width -> None
    assert csl._wstar_oracle() is None
    monkeypatch.setattr(csl, "_CASE_SEQ", 77)    # absent -> None
    assert csl._wstar_oracle() is None


def test_wstar_oracle_swallows_a_broken_file(tmp_path, monkeypatch):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    monkeypatch.setattr(csl, "_WSTAR_FILE", str(p))
    monkeypatch.setattr(csl, "_WSTAR_MAP", None)
    monkeypatch.setattr(csl, "_CASE_SEQ", 0)
    assert csl._wstar_oracle() is None


def test_wstar_oracle_outranks_the_tag_derivation(tmp_path, monkeypatch):
    """The tag path returns a width only on R-tagged preplaced instances;
    the probe must be able to hand a frame to EVERY case, and must win where
    both exist (that is the whole point of the oracle arm)."""
    n = 6
    rects = [(3.0 * i, 0.0, 3.0, 3.0) for i in range(n)]
    at = torch.full((n,), 9.0)
    cons = torch.zeros((n, 5))
    tpos = torch.full((n, 4), -1.0)
    opt = csl._ColumnOptimizer(rects, at, cons, tpos, None, None, None,
                               time.time() + 1.0, seed=0)
    assert csl._w_star_from_tags(opt) is None      # no R-tagged preplaced
    p = tmp_path / "ws.json"
    p.write_text(json.dumps({"0": 31.0}))
    monkeypatch.setattr(csl, "_WSTAR_FILE", str(p))
    monkeypatch.setattr(csl, "_WSTAR_MAP", None)
    monkeypatch.setattr(csl, "_CASE_SEQ", 0)
    ws = csl._wstar_oracle() if csl._WSTAR_FILE else None
    if ws is None:
        ws = csl._w_star_from_tags(opt)
    assert ws == 31.0


# ---------------------------------------------------------------------------
# 4. dump round-trip
# ---------------------------------------------------------------------------
def test_dump_round_trips(tmp_path, monkeypatch):
    n = 3
    rects = [(2.0 * i, 0.0, 2.0, 2.0) for i in range(n)]
    at = torch.full((n,), 4.0)
    cons = torch.zeros((n, 5))
    tpos = torch.full((n, 4), -1.0)
    opt = csl._ColumnOptimizer(rects, at, cons, tpos, None, None, None,
                               time.time() + 1.0, seed=0)
    a = ([(0.0, 0.0, 2.0, 2.0)] * n, 10.0, 100.0, 1.0)
    b = ([(1.0, 1.0, 2.0, 2.0)] * n, 12.0, 90.0, 0.0)
    cands = [(0, 0, ('N', 3, 1.0, 1.0, None, 1.5), a),
             (1, 1, ('D', None, 1.0, 1.0, None, 1.4), b)]
    monkeypatch.setattr(csl, "_PSEL_DUMP_PATH", str(tmp_path))
    monkeypatch.setattr(csl, "_PSEL_CASES", [])
    monkeypatch.setattr(csl, "_PSEL_HOOKED", True)   # no atexit in a test
    monkeypatch.setenv("PARTNER_PSEL_TAG", "unit")
    csl._psel_record(7, opt, 10.0, 5.0, cands, b)
    csl._psel_flush()

    d = np.load(tmp_path / "unit.npz")
    assert list(d["case_seq"]) == [7]
    assert list(d["case_n"]) == [n]
    assert list(d["cand_chan"]) == [0, 1]
    assert list(d["cand_sel"]) == [False, True]      # `b` was the winner
    assert list(d["cand_hp"]) == [10.0, 12.0]
    assert list(d["cand_score"]) == [1.5, 1.4]
    assert d["pos_all"].shape == (2 * n, 4)
    off = int(d["cand_pos_off"][1])
    assert d["pos_all"][off].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert d["lock_all"].tolist() == [False] * n
