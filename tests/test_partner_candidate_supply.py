import numpy as np

from candidate_supply_claude import CandidateBatch, allocate_quotas


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
