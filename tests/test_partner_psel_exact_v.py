"""Two more opt-in partner flags, both default off and both bit-exact when
off.

  * `PARTNER_PSEL_EXACT_V` -> the cross-restart champion arbitration in
    `column_sa_legalizer._parallel_solve` (`best_col`/`best_dir`) re-ranks
    each restart pool with `violation_killer._violations_exact` instead of
    the tolerant `full_violations` count `score()` normally reads off
    `o[3]`.  Implemented by `_arbitrate_champion_exact_v`: candidates are
    visited in increasing TOLERANT-score order, exact-scored, with an early
    exit floored at `min(4, len(pool))` visits.
  * `PARTNER_PICK_SCORE_REPAIRED` -> `contest_optimizer.MyOptimizer._pick_best`
    scores each direct candidate on its `_ensure_no_overlap`-repaired
    geometry instead of its raw geometry, and returns that already-repaired
    list on a direct win (no second repair pass).

What must hold:
  * flag off (either) -> byte-identical behavior on the affected path.
  * flag on (PARTNER_PSEL_EXACT_V) -> arbitration can select a candidate the
    tolerant-only ranking would not, and always exact-scores at least
    min(4, len(pool)) candidates.
  * flag on (PARTNER_PICK_SCORE_REPAIRED) -> the winning direct candidate's
    returned list reflects the repair, and the repair ran BEFORE scoring.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_ENV = ("PARTNER_PSEL_EXACT_V", "PARTNER_PICK_SCORE_REPAIRED",
        "PARTNER_PICK_EXACT_V")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)
    yield


# ---------------------------------------------------------------------------
# R1: PARTNER_PSEL_EXACT_V -- unit tests against the helper directly, since
# driving the real parallel-restart pool through solve() to manufacture a
# tolerant-undercount case is impractical.
# ---------------------------------------------------------------------------

def _make_score(hp_ref=1.0, area_ref=1.0, n_soft=1.0):
    def score(o):
        _out, hp, area, V = o
        return (1.0 + 0.5 * ((hp - hp_ref) / hp_ref
                             + max(0.0, area / area_ref - 1.0))) \
            * math.exp(2.0 * V / n_soft)
    return score


def test_psel_flag_off_does_not_import_violation_killer(monkeypatch):
    sys.modules.pop("violation_killer", None)
    monkeypatch.delenv("PARTNER_PSEL_EXACT_V", raising=False)
    import column_sa_legalizer as csa

    # off path never calls the new helper at all; just assert import
    # hygiene for the module under its default configuration.
    assert not csa._flag_on("PARTNER_PSEL_EXACT_V")
    assert "violation_killer" not in sys.modules


def test_arbitrate_champion_exact_v_overrides_tolerant_undercount(monkeypatch):
    import column_sa_legalizer as csa

    # A: tolerant counter misses a hidden violation (undercounts to 0), but
    # is exact-clean-checked as V=3 (a real grouping defect at a 1e-7 gap).
    # B: slightly worse tolerant proxy inputs, but truly clean (exact V=0).
    out_a = [(0.0, 0.0, 1.0, 1.0)]
    out_b = [(1.0, 0.0, 1.0, 1.0)]
    cand_a = (out_a, 10.0, 1.0, 0)     # tolerant score best (V=0 tolerant)
    cand_b = (out_b, 10.5, 1.0, 0)     # tolerant score slightly worse

    exact_map = {id(out_a): 3, id(out_b): 0}

    def fake_exact(_opt1, pos):
        # pos is np.asarray(out); match back to the source list via values
        if np.array_equal(pos, np.asarray(out_a)):
            return exact_map[id(out_a)]
        return exact_map[id(out_b)]

    monkeypatch.setitem(sys.modules, "violation_killer",
                         type(sys)("violation_killer"))
    sys.modules["violation_killer"]._violations_exact = fake_exact

    score = _make_score(hp_ref=10.0, n_soft=1.0)

    # sanity: tolerant-only argmin picks A
    assert min([cand_a, cand_b], key=score) is cand_a

    winner = csa._arbitrate_champion_exact_v([cand_a, cand_b], score,
                                             opt1=None, min_visits=4)
    assert winner is cand_b


def test_arbitrate_champion_exact_v_visits_floor_of_four(monkeypatch):
    import column_sa_legalizer as csa

    calls = []

    def fake_exact(_opt1, pos):
        calls.append(1)
        return 0

    monkeypatch.setitem(sys.modules, "violation_killer",
                         type(sys)("violation_killer"))
    sys.modules["violation_killer"]._violations_exact = fake_exact

    # 6 candidates, strictly increasing tolerant score, all exact-clean --
    # the "stop when next tolerant score exceeds current best" rule would
    # fire after 1 visit without the floor.
    pool = [([(0.0, 0.0, 1.0, 1.0)], 10.0 + i, 1.0, 0) for i in range(6)]
    score = _make_score(hp_ref=10.0, n_soft=1.0)

    winner = csa._arbitrate_champion_exact_v(pool, score, opt1=None,
                                             min_visits=4)
    assert winner is pool[0]
    assert len(calls) >= min(4, len(pool))


def test_arbitrate_champion_exact_v_empty_pool():
    import column_sa_legalizer as csa
    score = _make_score()
    assert csa._arbitrate_champion_exact_v([], score, opt1=None) is None


# ---------------------------------------------------------------------------
# R2: PARTNER_PICK_SCORE_REPAIRED -- unit-level against MyOptimizer._pick_best
# with a monkeypatched _ensure_no_overlap and a mocked scorer, per the
# existing partner test fixture pattern (see test_partner_pick_exact_v.py).
# ---------------------------------------------------------------------------

class _FakeScorer:
    n = 2
    kind = [0, 0]
    area_ref = 100.0
    n_soft_den = 1.0

    def _hpwl(self, pos):
        # marker: sum of x0 column identifies whether repair ran
        return float(pos[:, 0].sum())


def _bare_pick_best_optimizer():
    import contest_optimizer as co
    opt = co.MyOptimizer.__new__(co.MyOptimizer)
    opt.verbose = False
    opt._last_pick_channel = "column"
    return opt


def test_pick_score_repaired_off_repairs_only_after_selection(monkeypatch):
    import contest_optimizer as co

    monkeypatch.delenv("PARTNER_PICK_SCORE_REPAIRED", raising=False)
    calls = []

    def fake_repair(lst, locked):
        calls.append(list(lst))
        return [(x + 1000.0, y, w, h) for (x, y, w, h) in lst]

    monkeypatch.setattr(co, "_ensure_no_overlap", fake_repair)
    monkeypatch.setattr(co, "full_violations", lambda scorer, pos: 0)

    scorer = _FakeScorer()
    # force a clear direct win: column's raw hpwl (x0 sum) is much larger
    column_out = [(0.0, 0.0, 5.0, 5.0), (50.0, 0.0, 5.0, 5.0)]
    direct_out = [(0.0, 0.0, 5.0, 5.0), (1.0, 0.0, 5.0, 5.0)]
    box = [(direct_out, scorer)]

    opt = _bare_pick_best_optimizer()
    out = opt._pick_best(column_out, box)

    assert calls, "repair should have been invoked exactly once, post-selection"
    # regardless of which candidate won, repair must be called at most once
    # per _pick_best call in the off path (post-selection only)
    assert len(calls) <= 1


def test_pick_score_repaired_on_scores_repaired_geometry(monkeypatch):
    import contest_optimizer as co

    monkeypatch.setenv("PARTNER_PICK_SCORE_REPAIRED", "1")

    repair_calls = []
    scored_inputs = []

    def fake_repair(lst, locked):
        repair_calls.append(list(lst))
        # marker mutation: shift every rect's x by +1000
        return [(x + 1000.0, y, w, h) for (x, y, w, h) in lst]

    def fake_full_violations(scorer, pos):
        scored_inputs.append(np.array(pos, copy=True))
        return 0

    monkeypatch.setattr(co, "_ensure_no_overlap", fake_repair)
    monkeypatch.setattr(co, "full_violations", fake_full_violations)

    scorer = _FakeScorer()
    # make the direct candidate the clear winner even AFTER the +1000
    # repair-marker shift is applied to its geometry pre-scoring.
    column_out = [(1.0e6, 0.0, 5.0, 5.0), (1.0e6, 0.0, 5.0, 5.0)]
    direct_out = [(0.0, 0.0, 5.0, 5.0), (0.0, 0.0, 5.0, 5.0)]
    box = [(direct_out, scorer)]

    opt = _bare_pick_best_optimizer()
    out = opt._pick_best(column_out, box)

    assert repair_calls, "repair must run before scoring when flag is on"
    # the winning list must reflect the repair marker mutation
    assert all(x >= 1000.0 for (x, _y, _w, _h) in out)
    # the scorer must have been fed the repaired (shifted) geometry for the
    # direct candidate, not the raw pre-repair geometry
    direct_scored = [P for P in scored_inputs if P[:, 0].min() >= 1000.0]
    assert direct_scored, "scorer input for the direct candidate was not repaired"
