from floorset_arch.geometry import (
    Rect,
    bbox,
    boundary_satisfied,
    candidate_frontier_points,
    edge_touch_length,
    overlaps,
)


def test_touching_edges_are_not_overlap_and_have_touch_length():
    left = Rect(0.0, 0.0, 2.0, 3.0)
    right = Rect(2.0, 1.0, 4.0, 2.0)

    assert not overlaps(left, right)
    assert edge_touch_length(left, right) == 2.0


def test_boundary_bitmask_accepts_edges_and_corners():
    rects = [
        Rect(0.0, 0.0, 2.0, 3.0),
        Rect(2.0, 0.0, 4.0, 2.0),
        Rect(0.0, 3.0, 1.0, 1.0),
    ]
    box = bbox(rects)

    assert boundary_satisfied(rects[0], box, 1)
    assert boundary_satisfied(rects[2], box, 5)
    assert boundary_satisfied(rects[1], box, 10)
    assert not boundary_satisfied(rects[1], box, 4)


def test_candidate_frontier_points_include_existing_right_and_top_edges():
    placed = [Rect(0.0, 0.0, 2.0, 3.0), Rect(2.0, 0.0, 4.0, 2.0)]

    points = candidate_frontier_points(placed)

    assert (2.0, 0.0) in points
    assert (0.0, 3.0) in points
    assert (6.0, 0.0) in points

