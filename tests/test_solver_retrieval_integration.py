from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

CONTEST = Path(__file__).parents[1] / "FloorSet" / "iccad2026contest"
if str(CONTEST) not in sys.path:
    sys.path.insert(0, str(CONTEST))

import contest_optimizer
from candidate_supply import CandidateBatch
from contest_optimizer import FIRST_R4_RETRIEVAL_SLOTS, MyOptimizer, _select_ranked_source_quota
from retrieval_features import extract_retrieval_features
from retrieval_index import RetrievalResult


def _inputs():
    area = torch.tensor([4.0, 9.0])
    constraints = torch.tensor([[0.0, 1.0, 0, 0, 1], [1.0, 0.0, 0, 0, 2]])
    targets = torch.tensor([[7.0, 5.0, 2.0, 2.0], [-1.0, -1.0, 3.0, 3.0]])
    b2b = torch.tensor([[0.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    p2b = torch.tensor([[-1.0, -1.0, -1.0]])
    pins = torch.empty((0, 2))
    return area, constraints, targets, b2b, p2b, pins


def _optimizer(index=None, slots=2):
    optimizer = object.__new__(MyOptimizer)
    optimizer.retrieval_index = index
    optimizer.retrieval_slots = slots
    optimizer.retrieval_max_cost = 2.0
    optimizer.verbose = False
    return optimizer


def test_slots_zero_preserves_every_direct_slot():
    direct = [np.full((1, 4), float(i)) for i in range(4)]
    assert _select_ranked_source_quota(direct, ["direct"] * 4, [3, 2, 1, 0], 4, 0) == direct[::-1]


def test_ranked_quota_reserves_retrieval_and_direct_backfills():
    direct = [np.full((1, 4), 10.0), np.full((1, 4), 11.0), np.full((1, 4), 12.0)]
    retrieval = [np.full((1, 4), 20.0), np.full((1, 4), 21.0)]
    predictions = direct + retrieval
    selected = _select_ranked_source_quota(
        predictions, ["direct"] * 3 + ["retrieval"] * 2,
        [0, 1, 2, 3, 4], total=4, retrieval_quota=2,
    )
    assert [float(p[0, 0]) for p in selected] == [10.0, 11.0, 20.0, 21.0]
    selected = _select_ranked_source_quota(
        direct + retrieval[:1], ["direct"] * 3 + ["retrieval"],
        [3, 0, 1, 2], total=4, retrieval_quota=2,
    )
    assert [float(p[0, 0]) for p in selected] == [20.0, 10.0, 11.0, 12.0]


def test_missing_or_zero_retrieval_env_never_loads_index(monkeypatch):
    calls = []

    monkeypatch.setattr(MyOptimizer, "_load_model", lambda self: None)
    monkeypatch.setattr(MyOptimizer, "_load_direct_model", lambda self: None)
    monkeypatch.setattr(contest_optimizer, "init_worker_pool", lambda _workers: None)
    monkeypatch.setattr("retrieval_index.RetrievalIndex.load", lambda path: calls.append(path))
    monkeypatch.delenv("PARTNER_RETRIEVAL_INDEX", raising=False)
    monkeypatch.setenv("PARTNER_RETRIEVAL_SLOTS", "2")
    MyOptimizer(device="cpu")
    monkeypatch.setenv("PARTNER_RETRIEVAL_INDEX", "missing")
    monkeypatch.setenv("PARTNER_RETRIEVAL_SLOTS", "0")
    MyOptimizer(device="cpu")
    assert calls == []


def test_corrupt_index_fails_closed(monkeypatch):
    monkeypatch.setattr(MyOptimizer, "_load_model", lambda self: None)
    monkeypatch.setattr(MyOptimizer, "_load_direct_model", lambda self: None)
    monkeypatch.setattr(contest_optimizer, "init_worker_pool", lambda _workers: None)
    monkeypatch.setattr("retrieval_index.RetrievalIndex.load", lambda _path: (_ for _ in ()).throw(ValueError("bad")))
    monkeypatch.setenv("PARTNER_RETRIEVAL_INDEX", "corrupt")
    monkeypatch.setenv("PARTNER_RETRIEVAL_SLOTS", "2")
    optimizer = MyOptimizer(device="cpu")
    assert optimizer.retrieval_index is None
    assert optimizer.retrieval_slots == 0


def test_retrieval_env_slots_are_hard_capped_to_first_r4_capacity(monkeypatch):
    monkeypatch.setattr(MyOptimizer, "_load_model", lambda self: None)
    monkeypatch.setattr(MyOptimizer, "_load_direct_model", lambda self: None)
    monkeypatch.setattr(contest_optimizer, "init_worker_pool", lambda _workers: None)
    monkeypatch.setattr("retrieval_index.RetrievalIndex.load", lambda _path: object())
    monkeypatch.setenv("PARTNER_RETRIEVAL_INDEX", "pilot64")
    monkeypatch.setenv("PARTNER_RETRIEVAL_SLOTS", "5")
    optimizer = MyOptimizer(device="cpu")
    assert FIRST_R4_RETRIEVAL_SLOTS == 2
    assert optimizer.retrieval_slots == FIRST_R4_RETRIEVAL_SLOTS


def test_retrieval_matches_soft_d4_metadata_and_hard_anchors(monkeypatch):
    area, constraints, targets, b2b, p2b, pins = _inputs()
    features = extract_retrieval_features(
        area.numpy(), b2b.numpy(), p2b.numpy(), pins.numpy(), constraints.numpy(), targets.numpy()
    )
    index = SimpleNamespace(query=lambda *args, **kwargs: RetrievalResult(
        source_ids=np.array([71]), distances=np.array([0.25]),
        node_features=features.node_matrix[None],
        fp_xywh=np.array([[[0.0, 0.0, 3.0, 3.0], [4.0, 0.0, 2.0, 2.0]]]),
    ))
    seen = []
    # _sample_retrieval_preds imports the retrieval helpers lazily (the channel
    # is opt-in), so the spy has to live on the source module.
    import retrieval_transfer
    original = retrieval_transfer.remap_boundary_node_features
    monkeypatch.setattr(retrieval_transfer, "remap_boundary_node_features", lambda nodes, transform: (seen.append(transform), original(nodes, transform))[1])
    batch = _optimizer(index)._sample_retrieval_preds(2, area, constraints, targets, b2b, p2b, pins, 2)
    assert seen == ["identity", "mirror_x", "mirror_y", "transpose"]
    assert batch.source == "retrieval"
    assert len(batch.predictions) == 1
    assert batch.metadata["source_ids"] == [71]
    assert batch.metadata["retrieval_distances"] == [0.25]
    assert len(batch.metadata["transforms"]) == len(batch.predictions)
    assert np.isfinite(batch.predictions[0]).all()
    np.testing.assert_array_equal(batch.predictions[0][0], targets.numpy()[0])
    np.testing.assert_array_equal(batch.predictions[0][1, 2:], targets.numpy()[1, 2:])


def test_retrieval_empty_and_query_error_are_isolated():
    area, constraints, targets, b2b, p2b, pins = _inputs()
    for index in (SimpleNamespace(query=lambda *args, **kwargs: RetrievalResult(
            np.empty(0, dtype=np.int64), np.empty(0), np.empty((0, 2, 16)), np.empty((0, 2, 4)))),
                  SimpleNamespace(query=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("nope")))):
        batch = _optimizer(index)._sample_retrieval_preds(2, area, constraints, targets, b2b, p2b, pins, 2)
        assert batch.predictions == []
        assert batch.source == "retrieval"


def test_portfolio_disabled_is_direct_order_and_enabled_is_capacity_bounded(monkeypatch):
    optimizer = _optimizer(None, slots=0)
    direct = [np.full((1, 4), float(i + 1)) for i in range(4)]
    monkeypatch.setattr(optimizer, "_sample_direct_preds", lambda *args, **kwargs: direct)
    assert optimizer._sample_portfolio_preds(1, None, None, None, None, None, None, 3) == direct

    optimizer.retrieval_slots = 2
    optimizer.retrieval_index = object()
    monkeypatch.setattr(optimizer, "_sample_direct_raw_preds", lambda *args, **kwargs: direct)
    retrieved = CandidateBatch("retrieval", [np.full((1, 4), 9.0)], 0.0, {})
    monkeypatch.setattr(optimizer, "_sample_retrieval_preds", lambda *args, **kwargs: retrieved)
    monkeypatch.setattr(optimizer, "_rank_portfolio", lambda predictions, *_args: list(range(len(predictions))))
    got = optimizer._sample_portfolio_preds(1, np.array([1.0]), None, None, None, None, None, 3)
    assert len(got) == 3
    assert sum(float(p[0, 0]) == 9.0 for p in got) == 1


def test_portfolio_defensively_caps_env_slots_to_two_and_keeps_three_direct(monkeypatch):
    optimizer = _optimizer(object(), slots=5)
    direct = [np.full((1, 4), float(i + 1)) for i in range(5)]
    retrieved = CandidateBatch(
        "retrieval", [np.full((1, 4), 9.0), np.full((1, 4), 10.0)], 0.0, {}
    )
    requests = []
    monkeypatch.setattr(optimizer, "_sample_direct_raw_preds", lambda *args, **kwargs: direct)
    monkeypatch.setattr(
        optimizer, "_sample_retrieval_preds",
        lambda *args, **kwargs: (requests.append(args[-1]), retrieved)[1],
    )
    monkeypatch.setattr(optimizer, "_rank_portfolio", lambda predictions, *_args: list(range(len(predictions))))
    got = optimizer._sample_portfolio_preds(1, np.array([1.0]), None, None, None, None, None, 5)
    assert requests == [2]
    assert len(got) == 5
    assert sum(float(p[0, 0]) >= 9.0 for p in got) == 2
    assert sum(float(p[0, 0]) < 9.0 for p in got) == 3

