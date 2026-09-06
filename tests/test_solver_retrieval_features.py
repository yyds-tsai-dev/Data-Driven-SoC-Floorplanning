import numpy as np
import pytest

from retrieval_features import RetrievalFeatures, extract_retrieval_features


def _instance():
    area = np.array([4.0, 9.0, 16.0], dtype=np.float32)
    b2b = np.array([[0, 1, 2.0], [1, 2, 1.0], [-1, -1, -1]], dtype=np.float32)
    p2b = np.array([[0, 0, 1.0], [1, 2, 1.0], [-1, -1, -1]], dtype=np.float32)
    pins = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    constraints = np.array([
        [0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 2, 0, 2],
    ], dtype=np.float32)
    targets = np.array([[-1, -1, -1, -1], [-1, -1, 3, 3], [7, 2, 4, 4]], dtype=np.float32)
    return area, b2b, p2b, pins, constraints, targets


def test_feature_contract_is_finite_fixed_width_and_float32():
    features = extract_retrieval_features(*_instance())

    assert features.global_vector.shape == (24,)
    assert features.node_matrix.shape == (3, 16)
    assert features.global_vector.dtype == np.float32
    assert features.node_matrix.dtype == np.float32
    assert np.isfinite(features.global_vector).all()
    assert np.isfinite(features.node_matrix).all()


def test_direct_constructor_rejects_float64_arrays():
    with pytest.raises(ValueError, match="float32"):
        RetrievalFeatures(
            np.zeros(24, dtype=np.float64),
            np.zeros((3, 16), dtype=np.float64),
            3,
        )


def test_global_features_are_block_permutation_invariant_and_nodes_follow_blocks():
    area, b2b, p2b, pins, constraints, targets = _instance()
    permutation = np.array([2, 0, 1])
    inverse = np.argsort(permutation)

    b2b2 = b2b.copy()
    valid = b2b2[:, 0] >= 0
    b2b2[valid, :2] = inverse[b2b2[valid, :2].astype(int)]
    p2b2 = p2b.copy()
    valid = p2b2[:, 0] >= 0
    p2b2[valid, 1] = inverse[p2b2[valid, 1].astype(int)]
    got = extract_retrieval_features(
        area[permutation], b2b2, p2b2, pins, constraints[permutation], targets[permutation]
    )
    expected = extract_retrieval_features(area, b2b, p2b, pins, constraints, targets)

    np.testing.assert_allclose(got.global_vector, expected.global_vector, atol=1e-6)
    np.testing.assert_allclose(got.node_matrix, expected.node_matrix[permutation], atol=1e-6)
