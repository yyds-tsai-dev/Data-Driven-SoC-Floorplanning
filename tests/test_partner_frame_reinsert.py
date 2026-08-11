"""Preplaced-frame outlier reinsertion G0.

The pass may change topology, but only by removing movable rectangles beyond
an attainable preplaced tag line and placing them into legal slots inside the
resulting frame.  Dimensions and locked geometry are immutable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "partner") not in sys.path:
    sys.path.insert(0, str(ROOT / "partner"))

from frame_reinsert import (  # noqa: E402
    FrameReinsertResult,
    FrameTarget,
    frame_reinsert,
    frame_targets,
    reinsert_to_target,
    weighted_score,
)


def _top_fixture():
    # Block 2 is locked and tagged TOP, fixing y_max=4.  Block 1 overshoots to
    # y=6; the free lower-right slot [2,0]-[4,2] can receive it.
    rects = np.asarray([
        [0.0, 0.0, 2.0, 2.0],
        [0.0, 4.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 2.0],
    ])
    locked = np.asarray([False, False, True])
    boundary = np.asarray([0, 0, 4])
    cluster = np.zeros(3, dtype=np.int64)
    return rects, locked, boundary, cluster


def test_frame_targets_derives_overshot_preplaced_top_wall():
    rects, locked, boundary, _cluster = _top_fixture()

    got = frame_targets(rects, locked, boundary)

    assert got == [FrameTarget(axis=1, side=1, line=4.0, owner=2)]


def test_reinserts_movable_top_outlier_inside_preplaced_wall():
    rects, locked, _boundary, cluster = _top_fixture()
    target = FrameTarget(axis=1, side=1, line=4.0, owner=2)

    got = reinsert_to_target(
        rects,
        target,
        locked,
        cluster,
        lambda _p: 0.0,
    )

    assert got is not None
    assert np.max(got[:, 1] + got[:, 3]) <= 4.0 + 1e-9
    np.testing.assert_array_equal(got[:, 2:], rects[:, 2:])
    np.testing.assert_array_equal(got[locked], rects[locked])


def test_reinsertion_aborts_when_an_outlier_is_locked():
    rects, locked, _boundary, cluster = _top_fixture()
    locked[1] = True

    got = reinsert_to_target(
        rects,
        FrameTarget(axis=1, side=1, line=4.0, owner=2),
        locked,
        cluster,
        lambda _p: 0.0,
    )

    assert got is None


def test_reinsertion_aborts_when_an_outlier_belongs_to_cluster():
    rects, locked, _boundary, cluster = _top_fixture()
    cluster[1] = 7

    got = reinsert_to_target(
        rects,
        FrameTarget(axis=1, side=1, line=4.0, owner=2),
        locked,
        cluster,
        lambda _p: 0.0,
    )

    assert got is None


class _FakeOpt:
    def __init__(self):
        self.n = 3
        self.kind = [0, 0, 2]
        self.boundary = [0, 0, 4]
        self.cluster = [0, 0, 0]

    @staticmethod
    def _hpwl(_positions):
        return 0.0


def _top_violation(_opt, positions):
    p = np.asarray(positions)
    owner_top = p[2, 1] + p[2, 3]
    return int(np.max(p[:, 1] + p[:, 3]) > owner_top + 1e-6)


def test_guarded_pass_accepts_strict_v_drop_without_quality_regression():
    rects, _locked, _boundary, _cluster = _top_fixture()

    got = frame_reinsert(
        _FakeOpt(),
        [tuple(row) for row in rects],
        viol_fn=_top_violation,
        hpwl_fn=lambda _p: 0.0,
    )

    assert isinstance(got, FrameReinsertResult)
    assert got.attempted == 1
    assert got.accepted == 1
    assert _top_violation(_FakeOpt(), np.asarray(got.rects)) == 0


def test_guarded_pass_rejects_equal_violation_count():
    rects, _locked, _boundary, _cluster = _top_fixture()
    source = [tuple(row) for row in rects]

    got = frame_reinsert(
        _FakeOpt(), source, viol_fn=lambda _opt, _p: 1,
        hpwl_fn=lambda _p: 0.0,
    )

    assert got.accepted == 0
    assert got.rects is source
    assert got.reason == "no_monotone_candidate"


def test_guarded_pass_rejects_hpwl_regression():
    rects, _locked, _boundary, _cluster = _top_fixture()
    source = [tuple(row) for row in rects]

    got = frame_reinsert(
        _FakeOpt(), source, viol_fn=_top_violation,
        hpwl_fn=lambda p: -float(np.asarray(p)[1, 1]),
    )

    assert got.accepted == 0
    assert got.rects is source


def test_guarded_pass_contains_failures_and_returns_original_object():
    class _Broken:
        n = 3

        def __getattr__(self, _name):
            raise RuntimeError("broken")

    rects, _locked, _boundary, _cluster = _top_fixture()
    source = [tuple(row) for row in rects]
    got = frame_reinsert(_Broken(), source)

    assert got.rects is source
    assert got.accepted == 0
    assert got.reason == "exception"


def test_weighted_score_uses_full_case_denominator():
    counts = list(range(21, 121))
    before = [1.0] * 100
    after = before.copy()
    after[-1] = 0.9

    score_before = weighted_score(before, counts)
    score_after = weighted_score(after, counts)

    expected_weight_120 = 1.0 / sum(np.exp((n - 120) / 12.0) for n in counts)
    assert score_before == 1.0
    assert abs((score_before - score_after) - 0.1 * expected_weight_120) < 1e-12
