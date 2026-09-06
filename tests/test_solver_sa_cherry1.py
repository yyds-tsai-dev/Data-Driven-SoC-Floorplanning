"""Cherry-pick batch 1: the two src-only SA components ported into the
partner legalizer fork (`src/solver/column_sa_legalizer.py`).

1. `PARTNER_FASTSA_TEMP=1` -> Chen-Chang ISPD'05 three-stage cooling law
   (`_fastsa_temp`, a verbatim copy of the src helper) replaces the geometric
   `T = t0 * (t1/t0)**frac` curve inside `_anneal`.
2. `PARTNER_SA_STALL_STOP=1` -> the anneal chain breaks once best_cost has not
   improved by >= `stall_eps` (relative) within one stall window. The window
   is a fraction of the CURRENT chain's span (partner is deadline-bounded and
   has no inflated adaptive ceiling to rein in, unlike src, whose window is a
   fraction of the case base budget).

Both default OFF: with no env set the decision logic and the rng stream are
identical to the pre-port fork (the only additions are two branch predicates
per outer loop, which consume no randomness).
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

T0, T1 = 0.02, 0.0008           # the probe()/finish() anchors the port keeps
K, C, STEPS = 7.0, 100.0, 41.0  # paper (k=7, c=100) + our frac->index span

_ENV = ("PARTNER_FASTSA_TEMP", "PARTNER_FASTSA_K", "PARTNER_FASTSA_C",
        "PARTNER_FASTSA_STEPS", "PARTNER_SA_STALL_STOP",
        "PARTNER_SA_STALL_WINDOW", "PARTNER_SA_STALL_EPS")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


def _fs(frac):
    return lg._fastsa_temp(frac, T0, T1, K, C, STEPS)


def _geometric(frac):
    return T0 * (T1 / T0) ** frac


# --------------------------------------------------------------------------
# 1. temperature law invariants (pure function, dataset-free)
# --------------------------------------------------------------------------
def test_fastsa_endpoints_match_geometric_anchors():
    """Hot start == t0 and cold end == t1, so the accept regime the fork's
    cost scale is tuned for is preserved."""
    assert _fs(0.0) == pytest.approx(T0, abs=1e-12)
    assert _fs(1.0) == pytest.approx(T1, rel=1e-9)


def test_fastsa_stage2_is_a_deep_quench():
    """Stage 2 divides the stage-3 kernel by c, so it must sit far below the
    geometric law at the same frac."""
    frac = 0.05                      # n ~ 3, inside 2 <= n <= k
    assert _fs(frac) < _geometric(frac) / 50.0


def test_fastsa_reheats_at_n_equals_k():
    """The defining discontinuity: crossing n=k drops the /c factor, so T
    jumps up by ~c."""
    frac_k = (K - 1.0) / (STEPS - 1.0)
    before, after = _fs(frac_k - 1e-6), _fs(frac_k + 1e-6)
    assert after / before == pytest.approx(C, rel=1e-3)


def test_fastsa_stage3_is_monotone_one_over_n_tail():
    frac_k = (K - 1.0) / (STEPS - 1.0)
    xs = [frac_k + 1e-3 + i * 0.05 for i in range(18)]
    ts = [_fs(x) for x in xs if x <= 1.0]
    assert all(b < a for a, b in zip(ts, ts[1:]))


# --------------------------------------------------------------------------
# 2. env wiring on a real optimizer instance
# --------------------------------------------------------------------------
def _make_opt(**kw):
    """Minimal 3-soft-block optimizer; enough to exercise __init__ wiring and
    a short _anneal chain without the dataset."""
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


def test_defaults_are_off():
    opt = _make_opt()
    assert opt._sa_schedule == "geometric"
    assert opt._stall_frac == 0.0
    assert opt.stall_eps == 0.003


def test_fastsa_flag_and_param_overrides(monkeypatch):
    monkeypatch.setenv("PARTNER_FASTSA_TEMP", "1")
    monkeypatch.setenv("PARTNER_FASTSA_K", "9")
    monkeypatch.setenv("PARTNER_FASTSA_C", "50")
    monkeypatch.setenv("PARTNER_FASTSA_STEPS", "61")
    opt = _make_opt()
    assert opt._sa_schedule == "fastsa"
    assert (opt._fastsa_k, opt._fastsa_c, opt._fastsa_steps) == (9.0, 50.0, 61.0)


def test_fastsa_malformed_and_degenerate_params_fall_back(monkeypatch):
    monkeypatch.setenv("PARTNER_FASTSA_TEMP", "1")
    monkeypatch.setenv("PARTNER_FASTSA_K", "not-a-number")
    monkeypatch.setenv("PARTNER_FASTSA_C", "-3")
    monkeypatch.setenv("PARTNER_FASTSA_STEPS", "2")   # <= k, degenerate
    opt = _make_opt()
    assert opt._fastsa_k == 7.0
    assert opt._fastsa_c == 100.0
    assert opt._fastsa_steps > opt._fastsa_k


def test_stall_flag_and_overrides(monkeypatch):
    monkeypatch.setenv("PARTNER_SA_STALL_STOP", "1")
    monkeypatch.setenv("PARTNER_SA_STALL_WINDOW", "0.4")
    monkeypatch.setenv("PARTNER_SA_STALL_EPS", "0.01")
    opt = _make_opt()
    assert opt._stall_frac == 0.4
    assert opt.stall_eps == 0.01


def test_stall_window_out_of_range_falls_back(monkeypatch):
    monkeypatch.setenv("PARTNER_SA_STALL_STOP", "1")
    monkeypatch.setenv("PARTNER_SA_STALL_WINDOW", "1.5")   # not in (0, 1)
    monkeypatch.setenv("PARTNER_SA_STALL_EPS", "bogus")
    opt = _make_opt()
    assert opt._stall_frac == 0.25
    assert opt.stall_eps == 0.003


# --------------------------------------------------------------------------
# 3. stall stop actually terminates the chain early
# --------------------------------------------------------------------------
def test_stall_stop_breaks_before_the_deadline(monkeypatch):
    """A 3-block instance converges immediately, so with a short window the
    chain must exit well before its deadline and flag `_anneal_stalled`."""
    monkeypatch.setenv("PARTNER_SA_STALL_STOP", "1")
    monkeypatch.setenv("PARTNER_SA_STALL_WINDOW", "0.05")
    opt = _make_opt()
    opt.prepare()
    cols = opt._init_columns(opt.C0)
    cost, _ = opt._evaluate(cols)
    t_end = time.time() + 3.0
    opt._anneal(cols, t_end, cost)
    assert opt._anneal_stalled is True
    assert time.time() < t_end - 1.0


def test_stall_stop_off_runs_to_the_deadline():
    opt = _make_opt()
    opt.prepare()
    cols = opt._init_columns(opt.C0)
    cost, _ = opt._evaluate(cols)
    t_end = time.time() + 0.6
    opt._anneal(cols, t_end, cost)
    assert opt._anneal_stalled is False
    assert time.time() >= t_end
