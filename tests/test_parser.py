import torch

from floorset_arch.parser import parse_instance


def test_parse_instance_extracts_constraints_and_targets():
    areas = torch.tensor([100.0, 200.0, 300.0, -1.0])
    b2b = torch.tensor([[0.0, 1.0, 2.0], [-1.0, -1.0, -1.0]])
    p2b = torch.tensor([[0.0, 2.0, 1.5]])
    pins = torch.tensor([[10.0, 20.0]])
    constraints = torch.tensor(
        [
            [1.0, 0.0, 0.0, 2.0, 4.0],
            [0.0, 1.0, 1.0, 0.0, 10.0],
            [0.0, 0.0, 1.0, 2.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
        ]
    )
    targets = torch.tensor(
        [
            [-1.0, -1.0, 5.0, 20.0],
            [7.0, 8.0, 10.0, 20.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]
    )

    inst = parse_instance(3, areas, b2b, p2b, pins, constraints, targets)

    assert inst.block_count == 3
    assert inst.fixed == {0}
    assert inst.preplaced == {1}
    assert inst.mib_groups == {1: [1, 2]}
    assert inst.cluster_groups == {2: [0, 2]}
    assert inst.boundary[0] == 4
    assert inst.boundary[1] == 10
    assert inst.target_rects[0].width == 5.0
    assert inst.target_rects[1].x == 7.0
    assert inst.valid_b2b.shape == (1, 3)

