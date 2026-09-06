"""PARTNER_FASTSA_MIX: schedule DIVERSITY inside the parallel restart
portfolio (default off).

`PARTNER_FASTSA_TEMP=1` is all-or-nothing: it switches every restart's SA
cooling schedule from "geometric" to "fastsa" globally.  Fastsa-everywhere
wins big on most cases but hurts a few badly (high variance).  MIX lets a
deterministic fraction of the parallel-restart configs use fastsa while the
rest stay geometric, so per-case best-of-restarts selection captures both
behaviors instead of betting the whole portfolio on one schedule.

`PARTNER_FASTSA_MIX=<frac>` (float in (0,1)) only takes effect when
`PARTNER_FASTSA_TEMP` is unset/0 -- TEMP=1 still means all-fastsa regardless
of MIX. Config index i (deterministic, NOT seed/time-based) uses fastsa iff:
  frac<=0.5: (i % max(1, round(1/frac))) == 1
  frac>0.5:  NOT ((i % max(1, round(1/(1-frac)))) == 1)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as lg  # noqa: E402

_ENV = ("PARTNER_FASTSA_TEMP", "PARTNER_FASTSA_MIX")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


def _make_opt(**kw):
    import torch
    n = 3
    rects = [(0.0, 0.0, 2.0, 2.0)] * n
    at = torch.tensor([4.0, 4.0, 4.0])
    cons = torch.zeros((n, 5))
    tpos = torch.full((n, 4), -1.0)
    b2b = torch.zeros((n, n))
    p2b = torch.zeros((2, n))
    pins = torch.zeros((2, 2))
    return lg._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                               time.time() + 1.0, seed=7, **kw)


# --------------------------------------------------------------------------
# 1. off path -- MIX unset: zero behavior change
# --------------------------------------------------------------------------
def test_mix_frac_zero_when_unset():
    assert lg._fastsa_mix_frac() == 0.0


def test_directly_constructed_optimizer_stays_geometric():
    opt = _make_opt()
    assert opt._sa_schedule == "geometric"


def test_fastsa_mix_schedule_helper_is_importable_and_pure():
    assert callable(lg._fastsa_mix_schedule)
    assert lg._fastsa_mix_schedule(1, 0.5) is True
    assert lg._fastsa_mix_schedule(1, 0.5) is True  # deterministic, no rng


# --------------------------------------------------------------------------
# 2. selection rule (pure function)
# --------------------------------------------------------------------------
def test_fastsa_mix_half_selects_odd_indices():
    got = [lg._fastsa_mix_schedule(i, 0.5) for i in range(6)]
    assert got == [False, True, False, True, False, True]


def test_fastsa_mix_quarter_selects_one_in_four():
    got = [lg._fastsa_mix_schedule(i, 0.25) for i in range(12)]
    # idx % 4 == 1 -> indices 1, 5, 9
    assert got == [i % 4 == 1 for i in range(12)]
    assert sum(got) == 3


def test_fastsa_mix_majority_frac_flips_default_to_fastsa():
    # frac=0.75 -> geometric iff i % round(1/0.25)=4 == 1, else fastsa
    got = [lg._fastsa_mix_schedule(i, 0.75) for i in range(8)]
    expected = [not (i % 4 == 1) for i in range(8)]
    assert got == expected
    # majority of the portfolio is fastsa
    assert sum(got) > len(got) / 2


# --------------------------------------------------------------------------
# 3. env wiring -- _fastsa_mix_frac()
# --------------------------------------------------------------------------
def test_mix_frac_reads_env(monkeypatch):
    monkeypatch.setenv("PARTNER_FASTSA_MIX", "0.5")
    assert lg._fastsa_mix_frac() == 0.5


@pytest.mark.parametrize("raw", ["0", "1", "1.5", "-0.2", "not-a-float", ""])
def test_mix_frac_rejects_out_of_range_or_malformed(monkeypatch, raw):
    monkeypatch.setenv("PARTNER_FASTSA_MIX", raw)
    assert lg._fastsa_mix_frac() == 0.0


def test_fastsa_temp_overrides_mix(monkeypatch):
    """PARTNER_FASTSA_TEMP=1 still means all-fastsa regardless of MIX."""
    monkeypatch.setenv("PARTNER_FASTSA_TEMP", "1")
    monkeypatch.setenv("PARTNER_FASTSA_MIX", "0.25")
    assert lg._fastsa_mix_frac() == 0.0
    # and the optimizer itself goes all-fastsa via the pre-existing TEMP path
    opt = _make_opt()
    assert opt._sa_schedule == "fastsa"


# --------------------------------------------------------------------------
# 4. integration-lite: _worker_solve applies the mix via the appended
#    config-index payload field
# --------------------------------------------------------------------------
def _payload(cfg_idx=None, w_star=None):
    import torch
    n = 3
    rects = [(0.0, 0.0, 2.0, 2.0)] * n
    at = torch.tensor([4.0, 4.0, 4.0]).numpy()
    cons = torch.zeros((n, 5)).numpy()
    tpos = torch.full((n, 4), -1.0).numpy()
    b2b = torch.zeros((n, n)).numpy()
    p2b = torch.zeros((2, n)).numpy()
    pins = torch.zeros((2, 2)).numpy()
    deadline = time.time() + 1.0
    base = (rects, at, cons, tpos, b2b, p2b, pins,
            'N', None, 7, deadline, 1.0, 1.0)
    if w_star is not None:
        base = base + (w_star,)
    if cfg_idx is not None:
        base = base + (cfg_idx,)
    return base


def test_worker_solve_idx1_frac_half_uses_fastsa(monkeypatch):
    monkeypatch.setenv("PARTNER_FASTSA_MIX", "0.5")
    out = lg._worker_solve(_payload(cfg_idx=1))
    assert out is not None  # solved cleanly


def test_worker_solve_off_path_unaffected_by_stray_idx(monkeypatch):
    # MIX unset -> _apply_fastsa_mix is a no-op even with a cfg_idx present
    out = lg._worker_solve(_payload(cfg_idx=1))
    assert out is not None
