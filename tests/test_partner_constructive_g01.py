import numpy as np

from constructive_g01 import ConstructivePolicy, construct_candidate
from icdc.engine import verify_hard_legal


def _empty_edges() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


def test_construct_candidate_preserves_hard_constraints_and_avoids_obstacle():
    area = np.array([16.0, 9.0, 4.0, 6.0], dtype=np.float64)
    constraints = np.array(
        [
            [0, 1, 0, 0, 0],  # preplaced: origin and dimensions are immutable
            [1, 0, 0, 0, 0],  # fixed: dimensions are immutable
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    target_positions = np.array(
        [
            [2.0, 2.0, 4.0, 4.0],
            [-1.0, -1.0, 3.0, 3.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    policy = ConstructivePolicy(
        name="test",
        order_mode="large_first",
        bbox_weight=1.0,
        hpwl_weight=1.0,
        anchor_weight=0.0,
        constraint_weight=1.0,
    )

    result = construct_candidate(
        area,
        constraints,
        target_positions,
        _empty_edges(),
        _empty_edges(),
        np.empty((0, 2), dtype=np.float64),
        policy,
    )

    assert result.rects.shape == (4, 4)
    assert np.array_equal(result.rects[0], target_positions[0])
    assert np.array_equal(result.rects[1, 2:], target_positions[1, 2:])
    assert result.order[0] == 0
    assert result.hard_legal
    assert verify_hard_legal(result.rects, area, constraints, target_positions)["ok"]
