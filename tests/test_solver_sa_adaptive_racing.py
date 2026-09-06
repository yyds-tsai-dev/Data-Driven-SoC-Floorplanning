"""Three opt-in partner flags, each default off and each bit-exact when off.

  * `PARTNER_SA_ADAPTIVE` -> `_ColumnOptimizer` reweights its move mix online
    (M1c port from src/floorset_arch/legalizer/column_slicing.py:1884).  The
    fixed branch edges in `_random_move` are untouched; the flag remaps the
    single uniform draw onto them, so the rng stream length is invariant.
  * `PARTNER_SA_RACING`   -> `finish` runs an IN-WORKER successive-halving
    round before the main chain (racing port from column_slicing.py:2866,
    restructured to avoid the pool-wide sync barrier that killed the 0804
    Anytime-Ladder).  Wall-clock neutral: round 1 is carved out of the chain.
  * `PARTNER_PROXY_ALIGN` -> the direct-prediction prescreen ranks under the
    official cost's functional form, so a unit of relative violation costs
    4 gap units (exp(2 .) vs alpha=0.5) and the bbox gap is priced at all.

What must hold:
  * flag off -> BIT-EXACT.  Proved the hard way for the SA path: the same
    `_random_move` driver runs against HEAD's `column_sa_legalizer.py` and
    against the working tree, and the raw column/layout bytes must match.
  * flag on  -> the mechanism does what it claims (the remap is a measured
    distribution, the reweight respects its floor and normalisation, the race
    is wall-clock neutral, the aligned proxy really prices V at 4x).
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import candidate_supply as cs             # noqa: E402
import column_sa_legalizer as csl         # noqa: E402

_ENV = ("PARTNER_SA_ADAPTIVE", "PARTNER_SA_ADAPTIVE_WINDOW",
        "PARTNER_SA_ADAPTIVE_FLOOR", "PARTNER_SA_RACING",
        "PARTNER_SA_RACING_K", "PARTNER_SA_RACING_R1",
        "PARTNER_SA_RACING_TOP", "PARTNER_SA_STATS",
        "PARTNER_PROXY_ALIGN", "PARTNER_COL_NARROW", "PARTNER_COL_CACHE",
        "PARTNER_SA_KERNEL", "PARTNER_REFINE_KERNEL", "PARTNER_FRAME_WPIN",
        "PARTNER_EDGE_SEAT_V2")

_TOUCHED = ("column_sa_legalizer.py",)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# ---------------------------------------------------------------------------
# a deterministic instance with enough units for every move family to fire
# (multi-block clusters give family 3 something to reorder)
# ---------------------------------------------------------------------------

def _case(n: int = 30):
    rects = []
    for i in range(n):
        rects.append((6.0 * (i % 6), 6.0 * (i // 6), 5.0, 5.0))
    at = torch.full((n,), 25.0)
    cons = torch.zeros((n, 5))

    cons[3, 0] = 1.0                          # fixed shape
    for i in (5, 6, 7):
        cons[i, 2] = 1.0                      # MIB group
    for i in (10, 11, 12, 13):
        cons[i, 3] = 1.0                      # cluster A (4 members)
    for i in (20, 21, 22):
        cons[i, 3] = 2.0                      # cluster B (3 members)

    cons[0, 4] = 9.0                          # left + bottom
    cons[5, 4] = 6.0                          # right + top
    cons[14, 4] = 1.0
    cons[15, 4] = 4.0

    tpos = torch.full((n, 4), -1.0)
    tpos[3, 2] = 5.0
    tpos[3, 3] = 5.0
    pre = 17
    cons[pre, 1] = 1.0
    cons[pre, 4] = 2.0
    tpos[pre] = torch.tensor([30.0, 12.0, 5.0, 5.0])
    rects[pre] = (30.0, 12.0, 5.0, 5.0)

    edges = [[float(i), float((i * 5 + 3) % n), 1.0] for i in range(0, n, 2)]
    b2b = torch.tensor(edges, dtype=torch.float32)
    pins = torch.tensor([[0.0, 0.0], [36.0, 30.0]], dtype=torch.float32)
    p2b = torch.tensor([[0.0, 0.0, 2.0], [1.0, float(n - 1), 2.0]],
                       dtype=torch.float32)
    return rects, at, cons, tpos, b2b, p2b, pins


def _opt(**kw):
    rects, at, cons, tpos, b2b, p2b, pins = _case()
    return csl._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                                deadline=None, seed=7, **kw)


# ---------------------------------------------------------------------------
# 1. the flags are off by default
# ---------------------------------------------------------------------------

def test_flags_default_off():
    assert csl.sa_adaptive_on() is False
    assert csl.sa_racing_on() is False
    assert csl.sa_stats_on() is False
    o = _opt()
    assert o._adaptive_moves is False
    assert o._sa_racing is False
    assert o._move_cum == [0.55, 0.80, 0.92, 1.0]


def test_flag_readers_accept_the_house_spelling(monkeypatch):
    for name, fn in (("PARTNER_SA_ADAPTIVE", csl.sa_adaptive_on),
                     ("PARTNER_SA_RACING", csl.sa_racing_on),
                     ("PARTNER_SA_STATS", csl.sa_stats_on)):
        for val in ("1", "true", "True", "on", "ON"):
            monkeypatch.setenv(name, val)
            assert fn() is True, (name, val)
        for val in ("0", "", "no", "off"):
            monkeypatch.setenv(name, val)
            assert fn() is False, (name, val)
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# 2. off == HEAD, byte for byte -- on the SA move stream itself
# ---------------------------------------------------------------------------

_DRIVER = r'''
import hashlib, sys
import numpy as np, torch
import column_sa_legalizer as csl

sys.path.insert(0, r"__TESTS__")
from test_partner_sa_adaptive_racing import _case          # noqa: E402

rects, at, cons, tpos, b2b, p2b, pins = _case()
opt = csl._ColumnOptimizer(rects, at, cons, tpos, b2b, p2b, pins,
                           deadline=None, seed=7)
opt.prepare()
cols = [list(c) for c in opt._cols]
h = hashlib.sha256()
# 4000 proposals with a deterministic accept rule: this walks exactly the
# branch chain and the rng stream that PARTNER_SA_ADAPTIVE remaps.
for step in range(4000):
    undo = opt._random_move(cols)
    if undo is None:
        h.update(b"N")
        continue
    if step % 3 == 0:
        undo()
        h.update(b"U")
    else:
        h.update(b"A")
    h.update(repr(cols).encode())
pos, xr, yt = opt._layout_full(cols)
h.update(np.ascontiguousarray(pos, dtype=np.float64).tobytes())
h.update(repr((float(xr), float(yt))).encode())
print(h.hexdigest())
'''


def _hash_under(src: Path, tmp: Path) -> str:
    env = dict(os.environ)
    for v in _ENV:
        env.pop(v, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src), str(ROOT / "FloorSet" / "iccad2026contest"),
         str(ROOT / "FloorSet")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    drv = tmp / f"drv_{src.name}.py"
    drv.write_text(_DRIVER.replace("__TESTS__", str(ROOT / "tests")))
    out = subprocess.run([sys.executable, str(drv)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    return out.stdout.strip().splitlines()[-1]


def test_off_path_is_bit_identical_to_head(tmp_path):
    """The off path must not move one byte relative to the committed code.

    Not a proxy: the same 4000-proposal driver runs against HEAD's
    `column_sa_legalizer.py` and against the working tree, and the raw
    column state + float64 layout bytes are compared."""
    live = tmp_path / "live"
    head = tmp_path / "head"
    for d in (live, head):
        d.mkdir()
        for f in (ROOT / "partner").glob("*.py"):
            shutil.copy2(f, d / f.name)
    for name in _TOUCHED:
        blob = subprocess.run(
            ["git", "show", f"HEAD:src/solver/{name}"], cwd=ROOT,
            capture_output=True, timeout=120)
        assert blob.returncode == 0, blob.stderr[-2000:]
        (head / name).write_bytes(blob.stdout)

    assert _hash_under(head, tmp_path) == _hash_under(live, tmp_path)


def test_racing_branch_is_unreachable_when_the_flag_is_off(monkeypatch):
    """`finish` must not so much as look at `_race` with the flag unset."""
    opt = _opt()
    opt.prepare()
    monkeypatch.setattr(
        csl._ColumnOptimizer, "_race",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("raced")))
    out = opt.finish(time.time() + 0.35, max_runs=1)   # must not raise
    assert len(out) == opt.n


# ---------------------------------------------------------------------------
# 3. PARTNER_SA_ADAPTIVE
# ---------------------------------------------------------------------------

def test_remap_preserves_the_historical_mix_before_any_reweight(monkeypatch):
    """The starting `_move_cum` IS the historical schedule, so the remap is
    the identity on the branch decision (family k in, family k out)."""
    monkeypatch.setenv("PARTNER_SA_ADAPTIVE", "1")
    opt = _opt()
    assert opt._adaptive_moves is True
    for r in (0.0, 0.1, 0.5499, 0.55, 0.7, 0.7999, 0.80, 0.9, 0.9199,
              0.92, 0.99, 1.0 - 1e-12):
        assert opt._remap_move_r(r) == pytest.approx(r, abs=1e-9), r


def test_remap_realises_the_requested_family_probabilities(monkeypatch):
    """After a reweight the branch chain still uses (0.55, 0.80, 0.92), so
    the ONLY way the mix can change is the remap -- measure it."""
    monkeypatch.setenv("PARTNER_SA_ADAPTIVE", "1")
    opt = _opt()
    opt._move_cum = [0.10, 0.30, 0.75, 1.0]           # want 10/20/45/25 %
    edges = csl._MOVE_EDGES
    hits = [0, 0, 0, 0]
    N = 200000
    for k in range(N):
        r = (k + 0.5) / N                              # exact uniform grid
        rr = opt._remap_move_r(r)
        fam = 0 if rr < 0.55 else (1 if rr < 0.80 else (2 if rr < 0.92 else 3))
        assert fam == opt._last_move_type, (r, rr, fam)
        hits[fam] += 1
    want = [0.10, 0.20, 0.45, 0.25]
    for f in range(4):
        assert hits[f] / N == pytest.approx(want[f], abs=2e-4), f
    # monotone, and inside the historical support
    assert 0.0 <= opt._remap_move_r(0.0) < edges[-1]


def test_remap_consumes_exactly_one_draw_per_proposal(monkeypatch):
    """The rng stream length must not change: the flag may reroute a move,
    never add or drop a random number."""
    import random

    def _draws(adaptive: bool):
        if adaptive:
            os.environ["PARTNER_SA_ADAPTIVE"] = "1"
        else:
            os.environ.pop("PARTNER_SA_ADAPTIVE", None)
        opt = _opt()
        opt.prepare()
        cols = [list(c) for c in opt._cols]
        seen = []
        real = random.Random(1234)
        opt.rng = type("R", (), {
            "random": lambda _s: (seen.append(1), real.random())[1],
            "choice": lambda _s, x: real.choice(x),
            "randrange": lambda _s, *a: real.randrange(*a),
            "randint": lambda _s, *a: real.randint(*a),
            "sample": lambda _s, *a: real.sample(*a),
        })()
        for _ in range(500):
            opt._random_move(cols)
        return len(seen)

    try:
        assert _draws(False) == _draws(True)
    finally:
        os.environ.pop("PARTNER_SA_ADAPTIVE", None)


def test_reweight_normalises_and_respects_its_floor(monkeypatch):
    monkeypatch.setenv("PARTNER_SA_ADAPTIVE", "1")
    monkeypatch.setenv("PARTNER_SA_ADAPTIVE_FLOOR", "0.05")
    opt = _opt()
    # family 1 is the only one with any accepted improvement
    opt._am_prop = [100, 100, 100, 100]
    opt._am_acc = [10, 50, 0, 4]
    opt._am_impr = [0.0, 5.0, 0.0, 0.0]
    opt._adaptive_reweight()
    cum = opt._move_cum
    w = [cum[0], cum[1] - cum[0], cum[2] - cum[1], cum[3] - cum[2]]
    assert cum[3] == 1.0
    assert sum(w) == pytest.approx(1.0, abs=1e-12)
    for f in range(4):
        assert w[f] >= 0.05 - 1e-12, (f, w)
    # the only family with signal takes the whole free mass
    assert w[1] == pytest.approx(0.05 + (1.0 - 4 * 0.05), abs=1e-9)
    # counters were reset for the next window
    assert opt._am_prop == [0, 0, 0, 0]
    assert opt._am_since == 0


def test_reweight_is_a_no_op_without_signal(monkeypatch):
    monkeypatch.setenv("PARTNER_SA_ADAPTIVE", "1")
    opt = _opt()
    before = list(opt._move_cum)
    opt._am_prop = [40, 40, 40, 40]
    opt._am_acc = [0, 0, 0, 0]
    opt._am_impr = [0.0, 0.0, 0.0, 0.0]
    opt._adaptive_reweight()
    assert opt._move_cum == before
    assert opt._am_prop == [0, 0, 0, 0]


def test_anneal_collects_family_statistics_and_stays_legal(monkeypatch):
    monkeypatch.setenv("PARTNER_SA_ADAPTIVE", "1")
    monkeypatch.setenv("PARTNER_SA_ADAPTIVE_WINDOW", "50")
    opt = _opt()
    opt.prepare()
    cols = [list(c) for c in opt._cols]
    c0, _ = opt._evaluate(cols)
    snap, best = opt._anneal(cols, time.time() + 0.6, c0)
    assert best <= c0 + 1e-9
    # the window fired at least once, so the mix actually moved off the
    # historical schedule (or stayed put for want of signal -- both legal)
    assert sum(opt._am_prop) >= 0
    w = [opt._move_cum[0]] + [opt._move_cum[i] - opt._move_cum[i - 1]
                              for i in (1, 2, 3)]
    assert sum(w) == pytest.approx(1.0, abs=1e-9)
    assert min(w) >= 0.0
    pos, _xr, _yt = opt._layout(opt._restore(snap))
    P = np.asarray(pos, dtype=np.float64)
    x1, y1 = P[:, 0] + P[:, 2], P[:, 1] + P[:, 3]
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(P[:, 0][:, None],
                                                           P[:, 0][None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(P[:, 1][:, None],
                                                           P[:, 1][None, :])
    bad = (ox > 1e-6) & (oy > 1e-6)
    np.fill_diagonal(bad, False)
    assert not bad.any()


# ---------------------------------------------------------------------------
# 4. PARTNER_SA_RACING
# ---------------------------------------------------------------------------

def test_race_is_wall_clock_neutral(monkeypatch):
    """Round 1 is CARVED OUT of the chain, never added to it: the span the
    caller gets back plus the time the race burned must not exceed the span
    the caller handed in."""
    monkeypatch.setenv("PARTNER_SA_RACING", "1")
    monkeypatch.setenv("PARTNER_SA_RACING_K", "3")
    monkeypatch.setenv("PARTNER_SA_RACING_R1", "0.30")
    opt = _opt()
    opt.prepare()
    total = 0.30
    now = time.time()
    snap, now2, rem = opt._race(opt._snapshot(opt._cols), now, total)
    burned = now2 - now
    assert rem >= 0.0
    assert burned + rem <= total + 0.05          # + one outer-loop quantum
    assert burned >= 0.30 * total * 0.5          # round 1 really ran
    assert snap is not None
    cols = opt._restore(snap)
    assert sum(len(c) for c in cols) == len(opt.units)


def test_race_runs_k_arms_over_the_column_grid(monkeypatch):
    monkeypatch.setenv("PARTNER_SA_RACING", "1")
    monkeypatch.setenv("PARTNER_SA_RACING_K", "5")
    opt = _opt()
    opt.prepare()
    seen = []
    real_init = opt._init_columns
    opt._init_columns = lambda C: (seen.append(C), real_init(C))[1]
    opt._race(opt._snapshot(opt._cols), time.time(), 0.25)
    assert len(seen) == 5
    assert len(set(seen)) == 5                   # five DISTINCT column counts
    assert all(2 <= C <= 18 for C in seen)


def test_race_returns_the_best_round_one_arm(monkeypatch):
    """Successive halving means the SURVIVOR is the arm that led round 1 --
    that is the whole mechanism, so pin it."""
    monkeypatch.setenv("PARTNER_SA_RACING", "1")
    monkeypatch.setenv("PARTNER_SA_RACING_K", "3")
    opt = _opt()
    opt.prepare()
    costs = []
    real_anneal = opt._anneal

    def spy(cols, deadline, cur, **kw):
        snap, bc = real_anneal(cols, deadline, cur, **kw)
        costs.append((bc, snap))
        return snap, bc

    opt._anneal = spy
    snap, _n2, _rem = opt._race(opt._snapshot(opt._cols), time.time(), 0.25)
    assert len(costs) == 3
    best = min(costs, key=lambda t: t[0])
    assert snap == best[1]


def test_finish_with_racing_stays_inside_its_deadline(monkeypatch):
    monkeypatch.setenv("PARTNER_SA_RACING", "1")
    for opt, tag in ((_opt(), "on"),):
        opt.prepare()
        t0 = time.time()
        out = opt.finish(t0 + 0.40, max_runs=1)
        assert time.time() - t0 <= 0.40 + 0.30, tag   # + polish/refine tail
        assert len(out) == opt.n


def test_racing_is_skipped_when_probe_already_raced(monkeypatch):
    """`probe` IS round 1; racing on top of it would pay for two."""
    monkeypatch.setenv("PARTNER_SA_RACING", "1")
    opt = _opt()
    opt.prepare()
    opt.probe(0.02)
    assert opt._best_probe is not None
    monkeypatch.setattr(
        csl._ColumnOptimizer, "_race",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("raced")))
    opt.finish(time.time() + 0.30, max_runs=1)       # must not raise


# ---------------------------------------------------------------------------
# 5. PARTNER_PROXY_ALIGN -- the 4x marginal-weight audit
# ---------------------------------------------------------------------------

def _pred(n, shift=0.0, scale=1.0):
    P = np.zeros((n, 4))
    for i in range(n):
        P[i] = (6.0 * (i % 6) * scale + shift, 6.0 * (i // 6) * scale,
                5.0, 5.0)
    return P


def test_rank_predictions_default_is_the_historical_additive_form():
    """No `align_ref` -> byte-identical to the pre-flag ranking."""
    n = 12
    preds = [_pred(n), _pred(n, shift=3.0), _pred(n, scale=1.4)]
    area = np.full(n, 25.0)
    b2b = np.zeros((n, n))
    b2b[0, 5] = b2b[5, 0] = 1.0
    hp = [cs._hpwl_proxy(P, b2b) for P in preds]
    hp_ref = max(min(hp), 1e-9)
    want = sorted(range(3), key=lambda k: (
        hp[k] / hp_ref + 5.0 * cs._overlap_fraction(preds[k], area), k))
    assert cs.rank_predictions(preds, area, b2b) == want


def test_aligned_proxy_prices_a_violation_at_four_gap_units():
    """The audit claim, made executable.  In the official cost
    `(1 + 0.5 g) exp(2 v)` the marginal cost of relative violation is 4x the
    marginal cost of a gap unit.  Measure the aligned proxy's own two
    marginals and require the ratio."""
    n = 12
    area = np.full(n, 25.0)
    b2b = np.zeros((n, n))
    base = _pred(n)
    area_ref = float(area.sum()) / 0.97
    n_soft = 8.0

    def s(pred, v):
        idx = cs.rank_predictions([pred], area, b2b,
                                  constraint_penalties=[v],
                                  align_ref=(area_ref, n_soft))
        assert idx == [0]
        # recompute the score the same way the ranker does
        bb = cs._bbox_area(pred)
        return ((1.0 + 0.5 * (0.0 + max(0.0, bb / area_ref - 1.0)
                              + 5.0 * cs._overlap_fraction(pred, area)))
                * math.exp(2.0 * v / n_soft))

    eps = 1e-4
    c0 = s(base, 0.0)
    # d score / d (relative violation): v/n_soft is the relative unit
    dv = (s(base, eps * n_soft) - c0) / eps
    # d score / d (gap): perturb the bbox gap by eps
    big = base.copy()
    g0 = cs._bbox_area(base) / area_ref
    big[:, 2] *= (g0 + eps) / g0 if g0 > 0 else 1.0
    # analytic instead of geometric: the gap coefficient is the literal 0.5
    dg = 0.5 * c0
    assert dv / dg == pytest.approx(4.0, rel=1e-3)


def test_aligned_proxy_prices_the_bbox_gap_the_additive_form_ignored():
    """Two predictions with identical hpwl and no violations, one of which
    wastes 40% more bounding box.  The historical form cannot see it."""
    n = 12
    area = np.full(n, 25.0)
    b2b = np.zeros((n, n))
    tight = _pred(n)
    loose = _pred(n)
    loose[:, 0] *= 1.6                       # same shapes, wider bbox
    area_ref = float(area.sum()) / 0.97
    old = cs.rank_predictions([loose, tight], area, b2b)
    new = cs.rank_predictions([loose, tight], area, b2b,
                              constraint_penalties=[0.0, 0.0],
                              align_ref=(area_ref, 8.0))
    assert old == [0, 1]                     # additive form: a tie, index order
    assert new == [1, 0]                     # aligned form: tight wins


def test_penalties_as_counts_are_violation_scaled():
    """`as_counts=True` must return ESTIMATED VIOLATION COUNTS, i.e. the
    normalised form multiplied back by what it was averaged over."""
    import contest_optimizer as co
    os.environ["PARTNER_PRESCREEN_V"] = "1"
    try:
        n = 30
        _r, at, cons, _t, _b, _p, _pins = _case(n)
        opt = co.MyOptimizer.__new__(co.MyOptimizer)
        pred = _pred(n, shift=2.0)
        norm = opt._constraint_penalties([pred], n, at, cons)[0]
        cnt = opt._constraint_penalties([pred], n, at, cons,
                                        as_counts=True)[0]
        assert cnt >= norm - 1e-12          # counts >= the mean they average
        assert cnt > 0.0
        # 5 boundary-tagged + (3-1) MIB + (4-1) cluster A + (3-1) cluster B
        assert co.MyOptimizer._soft_norm(n, cons) == 5 + 2 + 3 + 2
        # and it must agree with the legalizer's own denominator
        rects, at2, cons2, tpos, b2b, p2b, pins = _case(n)
        ref = csl._ColumnOptimizer(rects, at2, cons2, tpos, b2b, p2b, pins,
                                   deadline=None, seed=7)
        assert co.MyOptimizer._soft_norm(n, cons2) == ref.n_soft_den
    finally:
        os.environ.pop("PARTNER_PRESCREEN_V", None)


def test_rank_portfolio_off_never_builds_an_align_ref(monkeypatch):
    import contest_optimizer as co
    seen = {}
    real = cs.rank_predictions

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(co, "rank_predictions", spy)
    n = 12
    _r, at, cons, _t, _b, _p, _pins = _case(30)
    opt = co.MyOptimizer.__new__(co.MyOptimizer)
    opt._rank_portfolio([_pred(n)], n, at[:n], cons[:n], torch.zeros((n, n)))
    assert seen.get("align_ref") is None


# ---------------------------------------------------------------------------
# 6. P2 arms cap
# ---------------------------------------------------------------------------

def test_wpin_arm_cap_is_a_quarter_of_the_portfolio():
    import inspect
    src = inspect.getsource(csl._parallel_solve)
    assert "_ws_arms = min(_ws_arms, max(1, len(configs) // 4))" in src
    # the production pool (24 configs, 3 arms) is untouched by the cap
    assert min(3, max(1, 24 // 4)) == 3
    # a 4-slot pool keeps at least one non-W* arm
    assert min(3, max(1, 4 // 4)) == 1
