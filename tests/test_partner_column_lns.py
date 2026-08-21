"""G0 tests for the default-off column split/merge large neighborhood."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "partner", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import column_sa_legalizer as csl  # noqa: E402
from column_lns import (  # noqa: E402
    best_split_merge_neighbor,
    iter_split_merge_neighbors,
    normalize_columns,
)


def _key(cols):
    return tuple(tuple(c) for c in cols)


def test_normalize_drops_empty_columns_without_mutating_input():
    cols = [[], [0, 1], [], [2], []]
    got = normalize_columns(cols, min_columns=2)
    assert got == [[0, 1], [2]]
    assert cols == [[], [0, 1], [], [2], []]


def test_normalize_keeps_minimum_column_slots():
    assert normalize_columns([[0], [], []], min_columns=2) == [[0], []]


def test_split_merge_neighbors_preserve_every_unit_exactly_once():
    cols = [[0, 1, 2], [3, 4], [5]]
    got = list(iter_split_merge_neighbors(cols, [None] * 6))

    assert got
    assert len({_key(c) for _kind, c in got}) == len(got)
    for kind, cand in got:
        assert kind in {"split", "merge"}
        assert sorted(k for col in cand for k in col) == list(range(6))
        assert all(col for col in cand)
        assert 2 <= len(cand) <= 18


def test_split_tries_both_left_right_orders_and_all_contiguous_cuts():
    got = {_key(c) for kind, c in iter_split_merge_neighbors(
        [[0, 1, 2], [3]], [None] * 4) if kind == "split"}

    assert ((0,), (1, 2), (3,)) in got
    assert ((1, 2), (0,), (3,)) in got
    assert ((0, 1), (2,), (3,)) in got
    assert ((2,), (0, 1), (3,)) in got


def test_merge_tries_both_vertical_orders():
    got = {_key(c) for kind, c in iter_split_merge_neighbors(
        [[0, 1], [2, 3], [4]], [None] * 5) if kind == "merge"}

    assert ((0, 1, 2, 3), (4,)) in got
    assert ((2, 3, 0, 1), (4,)) in got


def test_forced_left_and_right_units_stay_on_edge_columns():
    cols = [[0, 1, 2], [3, 4, 5]]
    forces = ["L", None, None, None, None, "R"]

    for _kind, cand in iter_split_merge_neighbors(cols, forces):
        where = {unit: ci for ci, col in enumerate(cand) for unit in col}
        assert where[0] == 0
        assert where[5] == len(cand) - 1


def test_max_and_min_column_limits_are_hard():
    at_max = [[i] for i in range(17)] + [[17, 18]]
    got_max = list(iter_split_merge_neighbors(at_max, [None] * 19,
                                              max_columns=18))
    assert all(kind != "split" for kind, _cand in got_max)

    got_min = list(iter_split_merge_neighbors([[0], [1]], [None, None],
                                              min_columns=2))
    assert all(kind != "merge" for kind, _cand in got_min)


class _FakeOptimizer:
    def __init__(self, scores):
        self.scores = scores

    def _evaluate(self, cols):
        return self.scores.get(_key(cols), 100.0), None


def test_oracle_selects_the_unique_improving_split():
    cols = [[0, 1, 2], [3]]
    target = ((0,), (1, 2), (3,))
    opt = _FakeOptimizer({_key(cols): 5.0, target: 1.0})

    result = best_split_merge_neighbor(opt, cols, [None] * 4)

    assert _key(result.cols) == target
    assert result.base_cost == pytest.approx(5.0)
    assert result.best_cost == pytest.approx(1.0)
    assert result.improved is True
    assert result.candidate_count > 0
    assert result.elapsed_s >= 0.0


def test_oracle_returns_normalized_identity_when_nothing_improves():
    cols = [[], [0, 1], [2], []]
    base = ((0, 1), (2,))
    opt = _FakeOptimizer({base: 2.0})

    result = best_split_merge_neighbor(opt, cols, [None] * 3)

    assert _key(result.cols) == base
    assert result.base_cost == result.best_cost == pytest.approx(2.0)
    assert result.improved is False


def test_oracle_flag_is_default_off(monkeypatch):
    monkeypatch.delenv("PARTNER_COL_LNS_ORACLE", raising=False)
    assert csl.col_lns_oracle_on() is False


def test_off_path_returns_same_columns_object(monkeypatch):
    monkeypatch.delenv("PARTNER_COL_LNS_ORACLE", raising=False)
    opt = object.__new__(csl._ColumnOptimizer)
    cols = [[0], [1]]
    assert opt._maybe_col_lns_oracle(cols) is cols
