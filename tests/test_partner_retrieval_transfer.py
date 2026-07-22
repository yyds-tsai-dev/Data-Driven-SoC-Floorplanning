import numpy as np
import pytest

from retrieval_matching_claude import hungarian_min_cost, match_blocks
from retrieval_transfer_claude import (
    apply_d4_rectangles,
    remap_boundary_node_features,
    transfer_layout,
)


def test_matching_recovers_area_and_constraint_correspondence():
    source = np.zeros((3, 16), dtype=np.float64)
    source[:, 0] = [0.0, 1.0, 2.0]
    source[1, 5] = 1.0
    source[2, 7] = 1.0
    target = source[[2, 0, 1]]

    result = match_blocks(source, target, max_cost=0.1)

    assert result.accepted
    assert result.target_to_source.tolist() == [2, 0, 1]


def test_transfer_preserves_target_area_and_hard_anchor():
    source = np.array([[0, 0, 2, 2], [3, 0, 3, 3]], dtype=np.float64)
    area = np.array([9.0, 4.0])
    constraints = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float64)
    target_positions = np.array([[7, 5, 3, 3], [-1, -1, -1, -1]], dtype=np.float64)

    got = transfer_layout(source, np.array([1, 0]), area, constraints, target_positions, "identity")

    np.testing.assert_allclose(got[:, 2] * got[:, 3], area, rtol=1e-6)
    np.testing.assert_allclose(got[0, :2], [7, 5], atol=0.0)


def test_hungarian_matches_brute_force_optimum_for_deterministic_costs():
    cost = np.array(
        [[9.0, 2.0, 7.0], [6.0, 4.0, 3.0], [5.0, 8.0, 1.0]], dtype=np.float64
    )
    assignment = hungarian_min_cost(cost)
    brute_force = min(
        sum(cost[row, column] for row, column in enumerate(permutation))
        for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    )

    assert cost[np.arange(3), assignment].sum() == brute_force


def test_matching_accepts_different_fixed_preplaced_mib_and_cluster_roles_below_threshold():
    source = np.zeros((2, 16), dtype=np.float64)
    source[0, 5:9] = [1.0, 0.0, 7.0, 3.0]
    source[1, 5:9] = [0.0, 1.0, 0.0, 0.0]
    target = source[::-1].copy()
    target[0, 5:9] = [0.0, 0.0, 2.0, 9.0]
    target[1, 5:9] = [1.0, 1.0, 4.0, 6.0]

    result = match_blocks(source, target, max_cost=10.0)

    assert result.accepted
    assert 0.0 < result.total_cost <= 10.0


def test_matching_accepts_extreme_finite_soft_feature_correspondence():
    source = np.zeros((2, 16), dtype=np.float64)
    source[:, 0] = [0.0, 8e7]
    target = np.zeros((2, 16), dtype=np.float64)
    target[:, 0] = [2e7, 1e8]

    result = match_blocks(source, target, max_cost=0.1)

    assert result.accepted
    assert result.target_to_source.tolist() == [0, 1]


def test_matching_single_block_has_a_confident_compatible_assignment():
    source = np.zeros((1, 16), dtype=np.float64)
    source[0, 0] = 2e7

    result = match_blocks(source, source.copy(), max_cost=0.0)

    assert result.accepted
    assert result.target_to_source.tolist() == [0]
    assert result.confidence == 1.0


def test_matching_equal_cost_ties_are_deterministic():
    source = np.zeros((3, 16), dtype=np.float64)

    result = match_blocks(source, source.copy(), max_cost=0.0)

    assert result.accepted
    assert result.target_to_source.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("transform", "expected_rectangles", "expected_boundaries"),
    [
        (
            "identity",
            [[1.0, 2.0, 3.0, 4.0], [6.0, 8.0, 2.0, 3.0]],
            [1.0, 2.0, 3.0, 4.0],
        ),
        (
            "mirror_x",
            [[5.0, 2.0, 3.0, 4.0], [1.0, 8.0, 2.0, 3.0]],
            [2.0, 1.0, 3.0, 4.0],
        ),
        (
            "mirror_y",
            [[1.0, 7.0, 3.0, 4.0], [6.0, 2.0, 2.0, 3.0]],
            [1.0, 2.0, 4.0, 3.0],
        ),
        (
            "transpose",
            [[2.0, 1.0, 4.0, 3.0], [8.0, 6.0, 3.0, 2.0]],
            [4.0, 3.0, 2.0, 1.0],
        ),
    ],
)
def test_d4_rectangles_and_boundary_features_use_the_same_direction(
    transform, expected_rectangles, expected_boundaries
):
    rectangles = np.array([[1, 2, 3, 4], [6, 8, 2, 3]], dtype=np.float64)
    node_features = np.zeros((1, 16), dtype=np.float64)
    node_features[0, 9:13] = [1, 2, 3, 4]

    np.testing.assert_allclose(apply_d4_rectangles(rectangles, transform), expected_rectangles)
    np.testing.assert_allclose(
        remap_boundary_node_features(node_features, transform)[0, 9:13], expected_boundaries
    )


@pytest.mark.parametrize("function", [apply_d4_rectangles, remap_boundary_node_features])
def test_d4_helpers_reject_unknown_transform(function):
    values = np.ones((1, 4 if function is apply_d4_rectangles else 16), dtype=np.float64)

    with pytest.raises(ValueError, match="unsupported D4 transform"):
        function(values, "rotate_90")


def test_role_mismatched_matching_then_transfer_overrides_target_fixed_and_preplaced_anchors():
    source = np.array([[0, 0, 2, 2], [3, 0, 2, 2]], dtype=np.float64)
    constraints = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float64)
    target_positions = np.array([[0, 0, 3, 5], [7, 11, 4, 6]], dtype=np.float64)
    source_nodes = np.zeros((2, 16), dtype=np.float64)
    source_nodes[:, 5:9] = [[0, 1, 3, 8], [1, 0, 0, 0]]
    target_nodes = np.zeros((2, 16), dtype=np.float64)
    target_nodes[:, 5:9] = [[1, 0, 9, 2], [0, 1, 5, 7]]
    match = match_blocks(source_nodes, target_nodes, max_cost=10.0)

    got = transfer_layout(
        source, match.target_to_source, np.array([4.0, 4.0]), constraints, target_positions, "identity"
    )

    assert match.accepted
    np.testing.assert_allclose(got[:, 2:4], [[3, 5], [4, 6]])
    np.testing.assert_allclose(got[1, :2], [7, 11], atol=0.0)


@pytest.mark.parametrize(
    "cost",
    [np.array([[0.0, np.nan]]), np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]])],
)
def test_hungarian_fails_closed_for_invalid_cost_matrices(cost):
    with pytest.raises(ValueError):
        hungarian_min_cost(cost)


def test_matching_fails_closed_for_nan_or_mismatched_features():
    source = np.zeros((2, 16), dtype=np.float64)
    target = np.zeros((3, 16), dtype=np.float64)

    with pytest.raises(ValueError):
        match_blocks(source, target, max_cost=1.0)
    source[0, 0] = np.nan
    with pytest.raises(ValueError):
        match_blocks(source, source.copy(), max_cost=1.0)


def test_transfer_requires_a_complete_equal_size_permutation():
    source = np.ones((3, 4), dtype=np.float64)
    constraints = np.zeros((2, 5), dtype=np.float64)
    target_positions = np.full((2, 4), -1.0, dtype=np.float64)

    with pytest.raises(ValueError, match="equal source and target block counts"):
        transfer_layout(
            source, np.array([0, 1]), np.array([1.0, 1.0]), constraints, target_positions, "identity"
        )
