import numpy as np

from candidate_supply_claude import CandidateBatch, allocate_quotas, rank_predictions


def test_candidate_batch_rejects_non_rectangles():
    bad = np.zeros((3, 3), dtype=np.float64)
    try:
        CandidateBatch("direct", [bad], 0.01, {})
    except ValueError as exc:
        assert "[N,4]" in str(exc)
    else:
        raise AssertionError("invalid candidate was accepted")


def test_allocate_quotas_never_expands_total_capacity():
    got = allocate_quotas(
        total=12,
        requested={"direct": 6, "retrieval": 3, "flow": 3},
        enabled=("direct", "retrieval", "flow"),
    )
    assert got == {"direct": 6, "retrieval": 3, "flow": 3}
    assert sum(got.values()) == 12


def test_allocate_quotas_preserves_direct_only_baseline():
    assert allocate_quotas(12, {"direct": 12}, ("direct",)) == {"direct": 12}


def test_rank_predictions_penalizes_overlap_after_hpwl_normalization():
    separated = np.array([[0, 0, 2, 2], [2, 0, 2, 2]], dtype=np.float64)
    overlapping = np.array([[0, 0, 2, 2], [1, 1, 2, 2]], dtype=np.float64)
    area = np.array([4.0, 4.0])
    b2b = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert rank_predictions([overlapping, separated], area, b2b) == [1, 0]


def test_direct_baseline_frozen_rank_order():
    predictions = [
        np.array([[0, 0, 2, 2], [1, 1, 2, 2]], dtype=np.float64),
        np.array([[0, 0, 2, 2], [2, 0, 2, 2]], dtype=np.float64),
        np.array([[0, 0, 2, 2], [8, 0, 2, 2]], dtype=np.float64),
    ]
    area = np.array([4.0, 4.0])
    b2b = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert rank_predictions(predictions, area, b2b) == [1, 0, 2]


def test_rank_predictions_accepts_source_neutral_constraint_penalties():
    p0 = np.array([[0, 0, 1, 1]], dtype=np.float64)
    p1 = np.array([[0, 0, 1, 1]], dtype=np.float64)
    order = rank_predictions(
        [p0, p1], np.array([1.0]), np.zeros((1, 1)),
        constraint_penalties=[2.0, 0.0], violation_weight=0.5,
    )
    assert order == [1, 0]


def test_rank_predictions_accepts_numpy_constraint_penalties():
    prediction = np.array([[0, 0, 1, 1]], dtype=np.float64)
    order = rank_predictions(
        [prediction, prediction], np.array([1.0]), np.zeros((1, 1)),
        constraint_penalties=np.array([1.0, 0.0]), violation_weight=1.0,
    )
    assert order == [1, 0]
