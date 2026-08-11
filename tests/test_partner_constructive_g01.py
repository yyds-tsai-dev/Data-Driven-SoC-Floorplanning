import numpy as np

from constructive_g01 import (
    ConstructivePolicy,
    construct_candidate,
    construct_portfolio,
    default_policies,
    encode_oracle_codes,
)
from icdc.engine import verify_hard_legal


def _empty_edges() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


def _policy(mode: str = "large_first") -> ConstructivePolicy:
    return ConstructivePolicy(
        name="test",
        order_mode=mode,
        bbox_weight=1.0,
        hpwl_weight=1.0,
        anchor_weight=0.0,
        constraint_weight=1.0,
    )


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
    result = construct_candidate(
        area,
        constraints,
        target_positions,
        _empty_edges(),
        _empty_edges(),
        np.empty((0, 2), dtype=np.float64),
        _policy(),
    )

    assert result.rects.shape == (4, 4)
    assert np.array_equal(result.rects[0], target_positions[0])
    assert np.array_equal(result.rects[1, 2:], target_positions[1, 2:])
    assert result.order[0] == 0
    assert result.hard_legal
    assert verify_hard_legal(result.rects, area, constraints, target_positions)["ok"]


def test_mib_shape_is_assigned_jointly_from_fixed_member():
    area = np.array([4.0, 4.0, 9.0], dtype=np.float64)
    constraints = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    target_positions = np.array(
        [
            [-1.0, -1.0, 4.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )

    result = construct_candidate(
        area,
        constraints,
        target_positions,
        _empty_edges(),
        _empty_edges(),
        np.empty((0, 2), dtype=np.float64),
        _policy(),
    )

    assert np.array_equal(result.rects[0, 2:], np.array([4.0, 1.0]))
    assert np.array_equal(result.rects[1, 2:], result.rects[0, 2:])
    assert result.hard_legal


def _edge_touches(left: np.ndarray, right: np.ndarray, tol: float = 1e-8) -> bool:
    x_overlap = min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0])
    y_overlap = min(left[1] + left[3], right[1] + right[3]) - max(left[1], right[1])
    vertical = abs(x_overlap) <= tol and y_overlap > tol
    horizontal = abs(y_overlap) <= tol and x_overlap > tol
    return vertical or horizontal


def test_cluster_members_are_scheduled_as_one_connected_frontier():
    area = np.array([4.0, 36.0, 4.0, 25.0, 4.0], dtype=np.float64)
    constraints = np.array(
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=np.float64,
    )
    target_positions = np.full((5, 4), -1.0, dtype=np.float64)

    result = construct_candidate(
        area,
        constraints,
        target_positions,
        _empty_edges(),
        _empty_edges(),
        np.empty((0, 2), dtype=np.float64),
        _policy("large_first"),
    )

    cluster_positions = sorted(result.order.index(block) for block in (0, 2, 4))
    assert cluster_positions == list(range(cluster_positions[0], cluster_positions[0] + 3))
    cluster = result.rects[[0, 2, 4]]
    adjacency = {
        i: {j for j in range(3) if i != j and _edge_touches(cluster[i], cluster[j])}
        for i in range(3)
    }
    reached = {0}
    frontier = [0]
    while frontier:
        unseen = adjacency[frontier.pop()] - reached
        reached.update(unseen)
        frontier.extend(unseen)
    assert reached == {0, 1, 2}
    assert result.diagnostics["disconnected_cluster_fallbacks"] == 0


def test_default_policies_are_stable_and_choose_distinct_orders():
    policies = default_policies()
    assert [policy.name for policy in policies] == [
        "net_closure",
        "constraint_first",
        "large_first",
        "pin_gravity",
    ]
    area = np.array([100.0, 4.0, 4.0, 9.0], dtype=np.float64)
    constraints = np.zeros((4, 5), dtype=np.float64)
    target_positions = np.full((4, 4), -1.0, dtype=np.float64)
    b2b = np.array([[1.0, 2.0, 20.0]], dtype=np.float64)
    p2b = np.array([[0.0, 3.0, 30.0]], dtype=np.float64)
    pins = np.array([[50.0, 50.0]], dtype=np.float64)

    results = construct_portfolio(
        area,
        constraints,
        target_positions,
        b2b,
        p2b,
        pins,
        policies=policies,
    )

    assert [result.policy for result in results] == [policy.name for policy in policies]
    assert len({result.order[0] for result in results}) >= 3
    assert all(result.hard_legal for result in results)


def test_oracle_region_encoding_discards_within_bin_coordinate_changes():
    first = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [15.0, 15.0, 1.0, 1.0],
            [4.10, 5.10, 2.0, 2.0],
        ],
        dtype=np.float64,
    )
    second = first.copy()
    second[2, :2] += 0.05

    code_a = encode_oracle_codes(first, region_bins=16)
    code_b = encode_oracle_codes(second, region_bins=16)

    assert np.array_equal(code_a.regions, code_b.regions)
    assert code_a.order == code_b.order
    assert np.issubdtype(code_a.regions.dtype, np.integer)
    assert not hasattr(code_a, "rects")
