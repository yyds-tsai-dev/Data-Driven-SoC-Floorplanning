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

from frame_reinsert import FrameTarget, frame_targets, reinsert_to_target  # noqa: E402


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
