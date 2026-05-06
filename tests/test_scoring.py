import torch

from floorset_arch import scoring
from floorset_arch.geometry import Rect
from floorset_arch.parser import parse_instance


def test_placement_score_uses_incident_edges_not_full_hpwl_proxy(monkeypatch):
    inst = parse_instance(
        4,
        torch.tensor([4.0, 4.0, 4.0, 4.0]),
        torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [2.0, 3.0, 999.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0, 3.0],
                [1.0, 3.0, 999.0],
            ]
        ),
        torch.tensor([[10.0, 0.0], [1000.0, 1000.0]]),
        torch.zeros(4, 5),
        None,
    )
    rects = {1: Rect(2.0, 0.0, 2.0, 2.0), 2: Rect(50.0, 50.0, 2.0, 2.0), 3: Rect(60.0, 60.0, 2.0, 2.0)}

    def full_scan_should_not_be_called(*_args, **_kwargs):
        raise AssertionError("placement_score must not scan all HPWL edges per candidate")

    monkeypatch.setattr(scoring, "hpwl_proxy", full_scan_should_not_be_called)

    score = scoring.placement_score(inst, 0, Rect(0.0, 0.0, 2.0, 2.0), rects)

    assert score < 1000.0

